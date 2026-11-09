#!/usr/bin/env python3
"""Generate behavior descriptions via the lab API, with CA validation and resumable cache."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import concurrent.futures as futures
import csv
from datetime import datetime
import fcntl
import hashlib
import itertools
import json
import os
import re
import random
import threading
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from output_layout import replace_if_changed
import sqlite3
import ssl
import sys
import time
import urllib.error
import urllib.request

BASE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017')
PROMPT = 'Please generate a short and concise behavior description based on the following API call chain. The description should primarily use nouns and verbs, maintaining consistent formatting and terminology.\n\nTo ensure the generated description is suitable for subsequent semantic similarity calculations, it is recommended to:\n\n- **Keep it concise and clear**: The description should be as brief as possible, highlighting key operations, and avoiding lengthy and complex sentence structures.\n- **Synthesize related functions**: Combine the functionalities of related functions to avoid repetitive or redundant descriptions.\n\nDo not use line breaks in your response; provide only the behavior description.'


def format_llm_response(text):
    if not isinstance(text,str):return ''
    text=re.sub(r'<think>.*?</think>','',text,flags=re.DOTALL|re.IGNORECASE)
    if re.search(r'</?think>',text,re.IGNORECASE):return ''
    text=text.strip().replace('```json','').replace('```','')
    if len(text)>=2 and text[0]==text[-1] and text[0] in ('"',"'"):text=text[1:-1]
    return ' '.join(text.split())


class Client:
    def __init__(self,args):
        if not args.base_url.startswith('https://'):raise ValueError('An HTTPS API URL is required')
        self.url=args.base_url.rstrip('/')
        self.model=args.model
        self.thinking='adaptive' if self.model=='MiniMax-M3.1-Flash' else 'disabled'
        self.reasoning_effort='low' if self.model=='MiniMax-M3.1-Flash' else None
        self.key=json.loads(args.api_config.read_text())['key']
        if not isinstance(self.key,str) or not self.key:raise ValueError('api.json has no valid key')
        self.context=ssl.create_default_context(cafile=str(args.ca_cert))
        self.opener=urllib.request.build_opener(urllib.request.HTTPSHandler(context=self.context),NoRedirect())
        self.timeout=args.timeout
        self.stop=threading.Event()
        self.metric_lock=threading.Lock()
        self.metrics=dict(request_attempts=0,request_errors=0,retry_attempts=0,retrying=0)
        self.fingerprint=hashlib.sha256(json.dumps([self.url,self.model,0,PROMPT,self.thinking,self.reasoning_effort,'retry-v2'],ensure_ascii=False).encode()).hexdigest()

    def cache_key(self,chain):
        return hashlib.sha256((self.fingerprint+'\0'+chain).encode()).hexdigest()

    def get_metrics(self):
        with self.metric_lock:return dict(self.metrics)

    def generate(self,chain):
        if not chain.strip():return 'blocked','','Empty call chain requires input repair'
        payload={'model':self.model,'temperature':0,'thinking':{'type':self.thinking},
                 'messages':[{'role':'user','content':PROMPT+'\nAPI call chain:\n'+chain}]}
        if self.reasoning_effort:payload['reasoning_effort']=self.reasoning_effort
        request=urllib.request.Request(self.url+'/chat/completions',data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},method='POST')
        attempt=0
        while not self.stop.is_set():
            attempt+=1;retry_after=0
            with self.metric_lock:
                self.metrics['request_attempts']+=1
                if attempt>1:self.metrics['retry_attempts']+=1
            try:
                if attempt>1:
                    retry_payload=dict(payload)
                    retry_payload['messages']=[{'role':'user','content':PROMPT+'\nReturn a concise description in English only, with at least five English words. Do not reproduce the call chain or use arrow symbols. Describe only the supplied operations.\nAPI call chain:\n'+chain}]
                    request=urllib.request.Request(self.url+'/chat/completions',data=json.dumps(retry_payload).encode(),
                        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},method='POST')
                with self.opener.open(request,timeout=self.timeout) as response:data=json.load(response)
                message=data['choices'][0]['message']
                raw=message.get('content','')
                # Some gateways return reasoning metadata even when disabled.
                # Keep only the final content; never persist reasoning fields.
                content=format_llm_response(raw)
                if len(content.split())<5 or any(x in content.lower() for x in ['failed to generate','error:','unable to process','->']):
                    raise ValueError('Empty or invalid behavior description')
                return 'success',content,''
            except urllib.error.HTTPError as error:
                detail=error.read(1500).decode(errors='replace').replace(self.key,'[redacted]')
                reason=f'HTTP {error.code}: {detail}'
                if error.code not in (408,409,425,429) and not 500<=error.code<=599:
                    return 'blocked','',reason
                try:retry_after=float(error.headers.get('Retry-After','0'))
                except (TypeError,ValueError):pass
            except (ssl.SSLCertVerificationError,):
                return 'blocked','','TLS certificate verification failed'
            except Exception as error:
                reason=f'{type(error).__name__}: {error}'.replace(self.key,'[redacted]')
                if isinstance(error,urllib.error.URLError) and isinstance(error.reason,ssl.SSLCertVerificationError):
                    return 'blocked','',reason
            with self.metric_lock:
                self.metrics['request_errors']+=1
                self.metrics['retrying']+=1
            delay=max(retry_after,min(60,2**min(attempt,6))+random.uniform(0,1))
            print(f'LLM retry {attempt}: wait {delay:.1f}s; chain={self.cache_key(chain)[:12]}; {reason[:220]}',flush=True)
            try:
                if self.stop.wait(delay):break
            finally:
                with self.metric_lock:self.metrics['retrying']-=1
        return 'cancelled','','Task stopped; record remains pending'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise urllib.error.HTTPError(req.full_url,code,'API redirects are disabled',headers,fp)


def write_status(root,data):
    data=dict(data,updated_at=time.time(),pid=os.getpid())
    target=root/'description_status.json';pending=target.with_suffix('.json.tmp')
    pending.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');pending.replace(target)


def prepare_plan(db,root,client,splits=('train','test')):
    db.execute('CREATE TABLE IF NOT EXISTS description_inputs (cache_key TEXT PRIMARY KEY,fingerprint TEXT,train_rows INT,test_rows INT)')
    db.execute('CREATE TABLE IF NOT EXISTS description_plan_meta (fingerprint TEXT PRIMARY KEY,signature TEXT)')
    db.commit()
    signature=json.dumps([(split,(root/split/'final_training_data.csv').stat().st_size,
        (root/split/'final_training_data.csv').stat().st_mtime_ns) for split in splits])
    existing=db.execute('SELECT signature FROM description_plan_meta WHERE fingerprint=?',(client.fingerprint,)).fetchone()
    if existing and existing[0]==signature:return
    counts=defaultdict(lambda:[0,0])
    for split in splits:
        index={'train':0,'test':1}[split]
        with (root/split/'final_training_data.csv').open(newline='',encoding='utf-8') as stream:
            for row in csv.DictReader(stream):counts[client.cache_key(row['call_chain'])][index]+=1
    with db:
        db.execute('DELETE FROM description_inputs WHERE fingerprint=?',(client.fingerprint,))
        db.executemany('INSERT INTO description_inputs VALUES (?,?,?,?)',
            ((key,client.fingerprint,*counts[key]) for key in sorted(counts)))
        db.execute('INSERT OR REPLACE INTO description_plan_meta VALUES (?,?)',(client.fingerprint,signature))
    print(f'PLAN: unique call chains={len(counts)}, train rows={sum(v[0] for v in counts.values())}, test rows={sum(v[1] for v in counts.values())}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--base-url',default='https://10.137.144.32/v1')
    parser.add_argument('--model',default='MiniMax-M3')
    parser.add_argument('--api-config',type=Path,default=BASE/'api.json')
    parser.add_argument('--ca-cert',type=Path,default=BASE/'lab-root-ca.crt')
    parser.add_argument('--splits',nargs='+',choices=['train','test'],default=None)
    parser.add_argument('--reuse-cache',type=Path,help='Read-only successful description cache with identical request fingerprint')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--set-workers',type=int,help='Update the running worker limit at the next 1000-row boundary')
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--probe',type=int,default=0,help='Test N unique chains only, without publishing CSVs')
    args=parser.parse_args()
    if args.splits is None:args.splits=[s for s in ('train','test') if (args.output_root/s/'final_training_data.csv').exists()]
    if not 1<=args.workers<=64:parser.error('workers must be 1..64')
    if args.probe<0 or args.timeout<=0:parser.error('Invalid probe count or timeout')
    csv.field_size_limit(sys.maxsize)
    root=args.output_root.resolve()
    if args.set_workers is not None:
        if not 1<=args.set_workers<=64:parser.error('set-workers must be 1..64')
        with sqlite3.connect((root/'descriptions.sqlite').as_uri()+'?mode=rw',uri=True) as config:
            config.execute('INSERT OR REPLACE INTO description_config VALUES (?,?)',('workers',args.set_workers))
        print(f'Worker limit requested: {args.set_workers}; applies at the next chunk boundary')
        return
    lock=(root/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    client=Client(args)
    db=sqlite3.connect(root/'descriptions.sqlite',timeout=60)
    db.execute('PRAGMA cache_size=-131072')
    db.execute('CREATE TABLE IF NOT EXISTS descriptions (cache_key TEXT PRIMARY KEY,fingerprint TEXT,status TEXT,description TEXT,error TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS description_config (name TEXT PRIMARY KEY,value INT)')
    db.execute('INSERT OR REPLACE INTO description_config VALUES (?,?)',('workers',args.workers))
    db.commit()
    cache={key:(status,description,error) for key,status,description,error in db.execute(
        "SELECT cache_key,status,description,error FROM descriptions WHERE fingerprint=? AND status='success'",(client.fingerprint,))}
    reuse_db=None
    if args.reuse_cache:
        reuse_db=sqlite3.connect(args.reuse_cache.resolve().as_uri()+'?mode=ro',uri=True)
        reuse_db.execute('PRAGMA query_only=ON')
    if args.probe:
        selected={}
        for split in dict.fromkeys(args.splits):
            with (root/split/'final_training_data.csv').open(newline='',encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    chain=row['call_chain'];key=client.cache_key(chain)
                    if chain and key not in selected:selected[key]=chain
                    if len(selected)>=args.probe:break
            if len(selected)>=args.probe:break
        for key,chain in selected.items():
            result=cache.get(key) or client.generate(chain)
            db.execute('INSERT OR REPLACE INTO descriptions VALUES (?,?,?,?,?)',(key,client.fingerprint,*result));db.commit()
            print(json.dumps(dict(status=result[0],description=result[1],error=result[2]),ensure_ascii=False),flush=True)
            if result[0]!='success':raise RuntimeError('API probe failed; full generation was not started')
        print(f'PROBE PASSED: model={client.model}, temperature=0 requested, thinking={client.thinking}, reasoning_effort={client.reasoning_effort}, TLS verified, chains={len(selected)}',flush=True)
        return
    started=time.time();api_done=0;api_failed=0;reused=0;rows_done=0
    progress=dict(state='planning',model=client.model,temperature=0,workers=args.workers,started_at=started,fingerprint=client.fingerprint,thinking=client.thinking,reasoning_effort=client.reasoning_effort,retry_policy='retry transient failures until success')
    write_status(root,progress)
    prepare_plan(db,root,client,tuple(dict.fromkeys(args.splits)))
    progress.update(state='running')
    write_status(root,progress)
    with ExitStack() as stack:
        pool=stack.enter_context(futures.ThreadPoolExecutor(max_workers=args.workers))
        stack.callback(client.stop.set)
        for split in dict.fromkeys(args.splits):
            source=root/split/'final_training_data.csv'
            final=root/split/'final_training_data_with_desc.csv'
            result=root/split/'result_with_descriptions.csv'
            failed=root/split/'result_with_descriptions_fail.csv'
            targets=[final,result,failed];pending_paths=[p.with_suffix('.csv.tmp') for p in targets]
            success_rows=failed_rows=split_rows=0
            with source.open(newline='',encoding='utf-8') as input_stream, \
                 pending_paths[0].open('w',newline='',encoding='utf-8') as enriched, \
                 pending_paths[1].open('w',newline='',encoding='utf-8') as success, \
                 pending_paths[2].open('w',newline='',encoding='utf-8') as failures:
                reader=csv.DictReader(input_stream);header=reader.fieldnames
                if not header or 'id' not in header or 'call_chain' not in header:raise ValueError(f'Invalid input: {source}')
                output=csv.DictWriter(enriched,fieldnames=header+['generated_description','status']);output.writeheader()
                success_writer=csv.writer(success);success_writer.writerow(['id','call_chain','generated_description','status'])
                failure_writer=csv.writer(failures);failure_writer.writerow(['id','call_chain','generated_description'])
                def input_rows():
                    while True:
                        chunk=list(itertools.islice(reader,1000))
                        if not chunk:return
                        if reuse_db is not None:
                            keys=list({client.cache_key(row['call_chain']) for row in chunk}-cache.keys())
                            for start in range(0,len(keys),500):
                                part=keys[start:start+500];marks=','.join('?' for _ in part)
                                found=reuse_db.execute(f"SELECT cache_key,status,description,error FROM descriptions WHERE fingerprint=? AND status='success' AND cache_key IN ({marks})",[client.fingerprint,*part]).fetchall()
                                for key,status,description,error in found:cache[key]=(status,description,error)
                        yield from chunk
                def emit(row,value):
                    nonlocal success_rows,failed_rows,split_rows,rows_done
                    status,description,error=value
                    if status!='success':raise RuntimeError('Description not generated; record retained: '+error)
                    output.writerow(dict(row,generated_description=description,status=status))
                    success_writer.writerow([row['id'],row['call_chain'],description,status])
                    success_rows+=1;split_rows+=1;rows_done+=1
                stream=iter(input_rows());todo={};waiting={};exhausted=False
                while todo or not exhausted:
                    # Bound submitted requests while continuously consuming new input.
                    # Duplicate chains share a future, IDs remain unchanged even if rows finish out of order.
                    while not exhausted and len(todo)<args.workers*2:
                        try:row=next(stream)
                        except StopIteration:exhausted=True;break
                        if not row['call_chain']:raise RuntimeError('Empty call chain; input repair required')
                        key=client.cache_key(row['call_chain'])
                        if key in cache:reused+=1;emit(row,cache[key])
                        elif key in waiting:waiting[key].append(row)
                        else:
                            waiting[key]=[row];todo[pool.submit(client.generate,row['call_chain'])]=key
                    if todo:
                        ready,_=futures.wait(todo,timeout=3,return_when=futures.FIRST_COMPLETED)
                        for future in ready:
                            key=todo.pop(future);value=future.result()
                            if value[0]!='success':
                                client.stop.set()
                                for queued in todo:queued.cancel()
                                progress.update(state='blocked',error=value[2],**client.get_metrics());write_status(root,progress)
                                raise RuntimeError('Description not generated; record retained: '+value[2])
                            cache[key]=value
                            db.execute('INSERT OR REPLACE INTO descriptions VALUES (?,?,?,?,?)',(key,client.fingerprint,*value));db.commit();api_done+=1
                            for row in waiting.pop(key):emit(row,value)
                    enriched.flush();success.flush();failures.flush()
                    progress.update(split=split,rows_done=rows_done,split_rows=split_rows,success_rows=success_rows,failed_rows=failed_rows,
                                    api_done=api_done,api_failed=api_failed,reused_rows=reused,inflight_or_queued=len(todo),**client.get_metrics())
                    write_status(root,progress)
            if failed_rows:raise RuntimeError('Incomplete descriptions; outputs were not published')
            for pending,target in zip(pending_paths,targets):replace_if_changed(pending,target)
            print(f'{split} PUBLISHED: {final}',flush=True)
    progress.update(state='finished',rows_done=rows_done,api_done=api_done,api_failed=api_failed,**client.get_metrics())
    write_status(root,progress)
    print('Behavior descriptions finished. No embedding, clustering or training was run.',flush=True)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from output_layout import dispatch,TRAIN
    dispatch(__file__,'--output-root',TRAIN,year_flag=None,training=False)
    main()
