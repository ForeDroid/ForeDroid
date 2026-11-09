#!/usr/bin/env python3
"""Extract GUI and call chains for selected years and classes."""
import argparse
import concurrent.futures as futures
import csv
from collections import defaultdict
import io
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

BATCHES = [(y, 'benign') for y in range(2014, 2018)]
BASE = Path(__file__).resolve().parent
from output_layout import raw_root
LOCK = threading.Lock()
POOL_LIMIT = 64


HEADER = ['icon','type','label','context','declaringClass','hascallback',
          'uid','container','rawtext','apk','id']
DONE = re.compile(r'^([0-9A-Fa-f]{64}): OCR enriched .*exported (\d+) UI rows')
FAILED = re.compile(r'GUI export failed for ([0-9A-Fa-f]{64}):')


def encode_rows(rows):
    stream = io.StringIO(newline='')
    csv.writer(stream, delimiter=';', quoting=csv.QUOTE_ALL).writerows(rows)
    return stream.getvalue()


def read_prefix(path):
    grouped = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream, delimiter=';', strict=True)
        if next(reader) != HEADER:
            raise ValueError(f'Unexpected GUI header: {path}')
        try:
            for row in reader:
                if len(row) == len(HEADER):
                    grouped[Path(row[9]).stem].append(row)
        except (csv.Error, UnicodeDecodeError):
            # A killed serial exporter can leave an incomplete final CSV record.
            # Reuse below requires an exact match with its logged row count.
            pass
    return grouped


def prepare(db, root, batches):
    db.execute('''CREATE TABLE IF NOT EXISTS gui_export_parts (
        year INT,label TEXT,apk TEXT,status TEXT,content TEXT,row_count INT,detail TEXT,
        PRIMARY KEY(year,label,apk))''')
    result = {}
    for year,label in sorted(batches):
        gui = raw_root(root)/str(year)/label/'gui'
        apps = sorted(p.parent for p in (gui/'output').glob('*/appSerialized.txt'))
        expected = {}; failures = set()
        log = gui/'logs/export.log'
        if log.exists():
            with log.open(errors='replace') as stream:
                for line in stream:
                    done = DONE.match(line); failed = FAILED.search(line)
                    if done: expected[done[1]] = int(done[2])
                    if failed: failures.add(failed[1])
        sources = sorted(gui.glob('.ocr-ui-*.tmp'), key=lambda p:p.stat().st_mtime)
        if (gui/'outputBenignUi_train.txt').exists():
            sources.append(gui/'outputBenignUi_train.txt')
        for source in sources:
            for apk,rows in read_prefix(source).items():
                if apk in expected and len(rows) == expected[apk] and apk not in failures:
                    db.execute('INSERT OR IGNORE INTO gui_export_parts VALUES (?,?,?,?,?,?,?)',
                               (year,label,apk,'done',encode_rows(rows),len(rows),'reused serial export'))
        for apk,count in expected.items():
            if count == 0 and apk not in failures:
                db.execute('INSERT OR IGNORE INTO gui_export_parts VALUES (?,?,?,?,?,?,?)',
                           (year,label,apk,'done','',0,'reused zero-row export'))
        for apk in failures:
            db.execute('INSERT OR IGNORE INTO gui_export_parts VALUES (?,?,?,?,?,?,?)',
                       (year,label,apk,'failed','',0,'previous export failure; not retried'))
        db.commit()
        result[year,label] = apps
        saved = db.execute('SELECT status,count(*) FROM gui_export_parts WHERE year=? AND label=? GROUP BY status',(year,label)).fetchall()
        print(f'{year}/{label}: models={len(apps)}, saved={saved}',flush=True)
    return result


def export_one(app, python):
    env = dict(os.environ, JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=2',
               OMP_THREAD_LIMIT='1', OCR_CONTINUE_ON_ERROR='0')
    with tempfile.TemporaryDirectory(prefix='foredroid-export-') as temporary:
        work = Path(temporary); inputs = work/'input'; inputs.mkdir()
        (inputs/app.name).symlink_to(app)
        (work/'res').symlink_to(BASE/'extract_Gui/res')
        output = work/'gui.txt'; log = work/'export.log'
        with log.open('wb') as stream:
            code = subprocess.run([str(BASE/'extract_Gui/tool/java8/bin/java'),'-Xmx8g',
                '-cp',f'{BASE}/extract_Gui/ocr/build:{BASE}/extract_Gui/jars/Backstage-UIAnalysis.jar',
                'OcrUIAnalysis',str(inputs),str(output),str(BASE/'extract_Gui/ocr/ocr_assets.py'),python],
                cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code or not output.exists():
            raise RuntimeError(f'Java exit {code}: '+log.read_text(errors='replace')[-2500:])
        rows = read_prefix(output)
        if any(apk != app.name for apk in rows):
            raise RuntimeError('Single-APK export contains another APK')
        records = rows.get(app.name,[])
        # Require the Java completion record, including when there are zero rows.
        matches = [DONE.match(line) for line in log.read_text(errors='replace').splitlines()]
        counts = [int(m[2]) for m in matches if m and m[1]==app.name]
        if counts != [len(records)]:
            raise RuntimeError('Single-APK export completion/row count mismatch')
        return encode_rows(records),len(records)


def publish_gui(db,root,year,label,apps):
    gui=raw_root(root)/str(year)/label/'gui'
    rows=db.execute('SELECT apk,status,content FROM gui_export_parts WHERE year=? AND label=? ORDER BY apk',(year,label)).fetchall()
    names={p.name for p in apps}
    assert names.issubset({row[0] for row in rows})
    fd,pending=tempfile.mkstemp(prefix='.gui-parallel-',suffix='.tmp',dir=gui)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream,delimiter=';',quoting=csv.QUOTE_ALL)
            writer.writerow(HEADER)
            row_id=0
            for apk,status,content in rows:
                if apk in names and status=='done':
                    for record in csv.reader(io.StringIO(content,newline=''),delimiter=';'):
                        record[-1]=str(row_id)
                        writer.writerow(record)
                        row_id+=1
        os.replace(pending,gui/'outputBenignUi_train.txt')
    finally:
        Path(pending).unlink(missing_ok=True)
    for path in gui.glob('.ocr-ui-*.tmp'):path.unlink()


def export_api(gui):
    target=gui/'api-results/api_train.txt'
    if target.is_file():return
    target.parent.mkdir(exist_ok=True)
    if not (gui/'res').exists():(gui/'res').symlink_to(BASE/'extract_Gui/res')
    with tempfile.TemporaryDirectory(prefix='foredroid-api-') as temporary, (gui/'logs/export.log').open('ab') as log:
        code=subprocess.run([str(BASE/'extract_Gui/tool/java8/bin/java'),'-Xmx8g','-jar',
            str(BASE/'extract_Gui/jars/Backstage-APIResultsProcessor.jar'),'-i',str(gui/'results'),'-o',temporary],
            cwd=gui,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=2')).returncode
        parts=sorted(Path(temporary).glob('api_*.txt'))
        if code or not parts:raise RuntimeError(f'API exporter exit {code}; files={len(parts)}')
        pending=target.with_suffix('.txt.tmp')
        try:
            with pending.open('w',newline='',encoding='utf-8') as output:
                writer=csv.writer(output,delimiter=';',quoting=csv.QUOTE_ALL);header=None
                for part in parts:
                    with part.open(newline='',encoding='utf-8') as stream:
                        reader=csv.reader(stream,delimiter=';');current=next(reader)
                        if header is None:header=current;writer.writerow(header)
                        if current!=header:raise RuntimeError('API header mismatch')
                        writer.writerows(reader)
            pending.replace(target)
        finally:pending.unlink(missing_ok=True)



def available_memory_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float('inf')


def call(command, cwd, log, timeout=None, env=None):
    with open(log, 'ab', buffering=0) as stream:
        p = subprocess.Popen([str(x) for x in command], cwd=cwd, stdout=stream,
                             stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            stream.write(b'\nAPK total timeout reached\n')
            return 124


def model_ok(path):
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        ET.parse(path)
        return True
    except (OSError, ET.ParseError):
        return False


def report_ok(path):
    try:
        return isinstance(json.loads(path.read_text()), list)
    except (OSError, ValueError):
        return False


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.root / 'run.lock', 'a+')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock_file.seek(0); self.lock_file.truncate()
        self.lock_file.write(str(os.getpid())); self.lock_file.flush()
        self.db = sqlite3.connect(self.root / 'state.sqlite', check_same_thread=False, timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (year INT, label TEXT, apk TEXT, module TEXT, status TEXT, seconds REAL, detail TEXT, PRIMARY KEY(year,label,apk,module))')
        self.db.execute('CREATE TABLE IF NOT EXISTS exports (year INT, label TEXT, status TEXT, detail TEXT, PRIMARY KEY(year,label))')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (year INT, label TEXT, apk TEXT, module TEXT, finished_at REAL, status TEXT, seconds REAL, PRIMARY KEY(year,label,apk,module))')
        self.db.execute('CREATE TABLE IF NOT EXISTS runtime_config (name TEXT PRIMARY KEY, value INT)')
        self.db.executemany('INSERT OR REPLACE INTO runtime_config VALUES (?,?)',
                            [('gui_workers',args.gui_workers),('chain_workers',args.chain_workers)])
        self.capacity = threading.Condition()
        self.active = {'gui':0, 'callchain':0}
        self.export_stats = {}
        self.export_started_at = None
        self.started_at = time.time()
        self.db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
        self.db.commit()
        self.java8 = BASE / 'extract_Gui/tool/java8/bin/java'
        self.java17 = shutil.which('java')
        if not self.java8.is_file() or not self.java17:
            raise RuntimeError('Missing Java 8/17 runtime')
        self.platforms = BASE / 'extract_callChain/lib/foredroid-platforms'
        self.platforms.mkdir(exist_ok=True)
        for api in range(18, 37):
            source = BASE / f'extract_Gui/android-18/android_jars/android-{api}.jar'
            if not source.is_file():
                raise RuntimeError(f'Missing Android platform {api}')
            folder = self.platforms / f'android-{api}'
            folder.mkdir(exist_ok=True)
            target = folder / 'android.jar'
            if not target.exists():
                target.symlink_to(source)
        self.env = dict(os.environ, JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=8')

    def refresh_limits(self):
        with LOCK:
            settings = dict(self.db.execute('SELECT name,value FROM runtime_config'))
        new = settings['gui_workers'], settings['chain_workers']
        if new != (self.args.gui_workers, self.args.chain_workers):
            with self.capacity:
                self.args.gui_workers, self.args.chain_workers = new
                self.capacity.notify_all()
            print(f'Live concurrency updated: GUI={new[0]}, callChain={new[1]}',flush=True)

    def acquire_slot(self, module):
        with self.capacity:
            while True:
                limit = self.args.gui_workers if module=='gui' else self.args.chain_workers
                if self.active[module] < limit and available_memory_gb() >= self.args.min_free_memory_gb:
                    self.active[module] += 1
                    return
                self.capacity.wait(timeout=3)

    def folder(self, year, label, module):
        path = raw_root(self.root) / str(year) / label / module
        for name in ['output', 'logs', 'results']:
            (path / name).mkdir(parents=True, exist_ok=True)
        return path

    def set_status(self, job, status, seconds=0, detail=''):
        year, label, apk, module = job
        with LOCK:
            self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?)',
                            (year, label, str(apk), module, status, seconds, detail))
            if status not in ('pending', 'running'):
                self.db.execute('INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)',
                                (year, label, str(apk), module, time.time(), status, seconds))
            self.db.commit()

    def gui(self, apk, folder, work, log):
        out = folder / 'output' / apk.stem
        if model_ok(out / 'appSerialized.txt'):
            return 'done', 'existing GUI model'
        code = call([self.java8, '-jar', BASE/'extract_Gui/tool/apktool_2.9.3.jar',
                     '-s', '-f', 'd', apk, '-o', out], work, log, 180, self.env)
        if code:
            return 'failed', f'apktool exit {code}'
        yml = (out/'apktool.yml').read_text()
        target = re.search(r'targetSdkVersion:\s*[\"\x27]?(\d+)', yml)
        minimum = re.search(r'minSdkVersion:\s*[\"\x27]?(\d+)', yml)
        api = int(target.group(1)) if target else int(minimum.group(1)) if minimum else 18
        if not target and api < 18:
            api = 28
        api = min(36, max(18, api))
        android = BASE/f'extract_Gui/android-18/android_jars/android-{api}.jar'
        code = call([self.java8, '-Xmx24g', '-Xss3m', '-jar', BASE/'extract_Gui/Backstage.jar',
                     '-apk', apk, '-androidJar', android, '-apkToolOutput', out,
                     '-logsDir', folder/'logs', '-resultsDir', folder/'results', '-rAnalysis',
                     '-uiTimeoutValue', '120', '-uiTimeoutUnit', 'SECONDS',
                     '-rTimeoutValue', '120', '-rTimeoutUnit', 'SECONDS',
                     '-maxDepthMethodLevel', '15', '-numThreads', '12', '-rLimitByPackageName'],
                    work, log, self.args.gui_timeout, self.env)
        if code == 124:
            return 'timeout', 'Backstage total timeout'
        if code:
            return 'failed', f'Backstage exit {code}'
        if not model_ok(out/'appSerialized.txt'):
            tail = log.read_text(errors='replace')[-50000:]
            if 'Non english language detected' in tail:
                return 'filtered', 'Backstage language filter'
            if 'less than 70%' in tail:
                return 'filtered', 'Backstage layout-count/activity-count filter'
            return 'failed', 'Backstage returned without a valid GUI model'
        # Preserve the same decoded resources retained by the original cleanup.
        keep = {'appSerialized.txt', 'images', 'AndroidManifest.xml', 'res', 'apktool.yml'}
        for p in out.iterdir():
            if p.name not in keep:
                if p.is_dir(): shutil.rmtree(p)
                else: p.unlink()
        return 'done', f'android-{api}'

    def chain(self, apk, folder, work, log):
        report = folder/'output'/apk.stem/'CallGraphInfo/rule_analysis_report.json'
        if report_ok(report):
            return 'done', 'existing call-chain report'
        code = call([self.java17, '-Xmx12g', '-jar', BASE/'extract_callChain/ICCBot.jar',
                     '-client', 'CTGClient', '-maxPathNumber', '100',
                     '-androidJar', self.platforms, '-path', apk.parent,
                     '-outputDir', folder/'output', '-name', apk.name, '-time', '30'],
                    work, log, self.args.chain_timeout, self.env)
        if code == 124:
            return 'timeout', 'ICCBot total timeout'
        if code:
            return 'failed', f'ICCBot exit {code}'
        if not report_ok(report):
            return 'failed', 'ICCBot returned without a valid JSON report'
        return 'done', 'call-chain report written'

    def run_job(self, job):
        year, label, apk, module = job
        folder = self.folder(year, label, module)
        log = folder/'logs'/f'{apk.stem}.log'
        self.acquire_slot(module)
        start = time.time()
        try:
            self.set_status(job, 'running')
            try:
                with tempfile.TemporaryDirectory(prefix=f'foredroid-{module}-') as tmp:
                    work = Path(tmp)
                    source = BASE / ('extract_Gui' if module == 'gui' else 'extract_callChain')
                    for name in (['res'] if module == 'gui' else ['resources', 'config']):
                        (work/name).symlink_to(source/name)
                    status, detail = self.gui(apk, folder, work, log) if module == 'gui' else self.chain(apk, folder, work, log)
            except Exception as error:
                status, detail = 'failed', str(error)
            self.set_status(job, status, time.time()-start, detail)
        finally:
            with self.capacity:
                self.active[module] -= 1
                self.capacity.notify_all()
        print(f'{year}/{label} {module} {apk.name}: {status} {detail}', flush=True)
        return year, label, module

    def export_app(self, app):
        while available_memory_gb() < self.args.min_free_memory_gb:
            time.sleep(3)
        return export_one(app, self.args.python)

    def export(self, year, label):
        # Each batch coordinator owns its SQLite connection; APK workers only
        # produce isolated temporary CSV data. One shared pool bounds all batches.
        folder = self.folder(year, label, 'gui')
        with sqlite3.connect(self.root/'state.sqlite', timeout=60) as db:
            apps = prepare(db, self.root, [(year,label)])[year,label]
            saved = dict(db.execute('SELECT apk,status FROM gui_export_parts WHERE year=? AND label=?',(year,label)))
            progress = dict(total=len(apps), done=sum(saved.get(a.name)=='done' for a in apps),
                            failed=sum(saved.get(a.name)=='failed' for a in apps),
                            reused=sum(saved.get(a.name)=='done' for a in apps),
                            remaining=sum(a.name not in saved for a in apps), processed_new=0)
            with LOCK:
                self.export_stats[year,label] = progress
                if self.export_started_at is None:
                    self.export_started_at = time.time()
            db.execute('INSERT OR REPLACE INTO exports VALUES (?,?,?,?)',(year,label,'running','parallel OCR/GUI export'))
            db.commit()
            detail = ''
            try:
                if not apps:
                    raise RuntimeError('No GUI models were produced')
                pending = {self.export_pool.submit(self.export_app,a):a for a in apps if a.name not in saved}
                for future in futures.as_completed(pending):
                    app = pending[future]
                    try:
                        content,count = future.result(); state='done'; error=''
                    except Exception as exc:
                        content=''; count=0; state='failed'; error=str(exc)
                    db.execute('INSERT OR REPLACE INTO gui_export_parts VALUES (?,?,?,?,?,?,?)',
                               (year,label,app.name,state,content,count,error))
                    db.commit()
                    with LOCK:
                        progress[state] += 1
                        progress['remaining'] -= 1
                        progress['processed_new'] += 1
                    with (folder/'logs/export.log').open('a') as log:
                        log.write(f'{app.name}: parallel export {state}, rows={count}\n')
                        if error: log.write(error+'\n')
                    print(f'{year}/{label} {app.name}: export {state}, rows={count}',flush=True)
                publish_gui(db,self.root,year,label,apps)
                export_api(folder)
                state='partial' if progress['failed'] else 'done'
                detail=f"{progress['failed']} exports failed; not retried" if progress['failed'] else ''
            except Exception as exc:
                state='failed'; detail=str(exc)
            db.execute('INSERT OR REPLACE INTO exports VALUES (?,?,?,?)',(year,label,state,detail))
            db.commit()
        print(f'{year}/{label} GUI/OCR/API export: {state} {detail}',flush=True)

    def snapshot(self, state):
        with LOCK:
            counts=self.db.execute('SELECT year,label,module,status,count(*) FROM jobs GROUP BY year,label,module,status').fetchall()
            exports=self.db.execute('SELECT year,label,status,detail FROM exports').fetchall()
            export_progress={key:sum(p[key] for p in self.export_stats.values()) for key in ('total','done','failed','remaining','reused')}
            elapsed=time.time()-self.export_started_at if self.export_started_at else 0
            processed=sum(p['processed_new'] for p in self.export_stats.values())
            eta=export_progress['remaining']*elapsed/processed if elapsed>=30 and processed else None
            export_progress.update(workers=self.args.export_workers,eta=eta,stage='finished' if state=='finished' else 'OCR/GUI/API')
            extracting=any(row[3] in ('pending','running') for row in counts)
        data={'state':state,'pid':os.getpid(),'updated_at':time.time(), 'started_at':self.started_at,
              'gui_workers':self.args.gui_workers,'chain_workers':self.args.chain_workers,
              'gui_timeout':self.args.gui_timeout,'chain_timeout':self.args.chain_timeout,
              'min_free_memory_gb':self.args.min_free_memory_gb,
              'phase':'extract' if extracting else 'export', 'export_progress':export_progress,
              'counts':[dict(zip(['year','label','module','status','count'],r)) for r in counts],
              'exports':[dict(zip(['year','label','status','detail'],r)) for r in exports]}
        target=self.root/'status.json'
        target.with_suffix('.json.tmp').write_text(json.dumps(data,indent=2)+'\n')
        target.with_suffix('.json.tmp').replace(target)

    def run(self):
        jobs=[]
        for year,label in self.args.batches:
            apks=sorted((Path(self.args.dataset)/str(year)/label).rglob('*.apk'))
            if not apks:raise RuntimeError(f'No APKs in {year}/{label}')
            for apk in apks:
                for module in ['gui','callchain']:
                    jobs.append((year,label,apk.resolve(),module))
        if self.args.pilot:
            jobs=[j for j in jobs if j[2].name.startswith(self.args.pilot)]
            if len(jobs)!=2:raise RuntimeError('Pilot must select exactly one APK')
        selected_batches = {(j[0], j[1]) for j in jobs}
        with LOCK:
            for y,l,a,m in jobs:
                self.db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?)',(y,l,str(a),m,'pending',0,''))
            self.db.commit()
            previous={(r[0],r[1],r[2],r[3]):r[4] for r in self.db.execute('SELECT year,label,apk,module,status FROM jobs')}
        jobs=[j for j in jobs if previous.get((j[0],j[1],str(j[2]),j[3])) not in ('done',) and
              (self.args.retry_failed or previous.get((j[0],j[1],str(j[2]),j[3])) in ('pending','running'))]
        print(f'Selected {len(jobs)} pending module jobs; output={self.root}',flush=True)
        self.snapshot('running')
        with futures.ThreadPoolExecutor(max_workers=POOL_LIMIT) as gui, \
             futures.ThreadPoolExecutor(max_workers=POOL_LIMIT) as chain, \
             futures.ThreadPoolExecutor(max_workers=self.args.export_workers) as self.export_pool, \
             futures.ThreadPoolExecutor(max_workers=len(self.args.batches)) as export:
            pending={ (gui if j[3]=='gui' else chain).submit(self.run_job,j) for j in jobs }
            exports=[];exported=set()
            while pending:
                self.refresh_limits()
                finished,pending=futures.wait(pending,timeout=30,return_when=futures.FIRST_COMPLETED)
                for f in finished:
                    y,l,module=f.result()
                    if module=='gui' and (y,l) not in exported:
                        with LOCK:
                            count=self.db.execute("SELECT count(*) FROM jobs WHERE year=? AND label=? AND module='gui' AND status IN ('pending','running')",(y,l)).fetchone()[0]
                        if count==0:
                            exported.add((y,l));exports.append(export.submit(self.export,y,l))
                self.snapshot('running')
            for y,l in sorted(selected_batches - exported):
                with LOCK:
                    previous_export=self.db.execute('SELECT status FROM exports WHERE year=? AND label=?',(y,l)).fetchone()
                if not previous_export or previous_export[0]=='running':
                    exports.append(export.submit(self.export,y,l))
            waiting=set(exports)
            while waiting:
                finished,waiting=futures.wait(waiting,timeout=3,return_when=futures.FIRST_COMPLETED)
                for f in finished:f.result()
                self.snapshot('running')
        self.snapshot('finished')
        print('Extraction finished; failures/filters/timeouts remain recorded in state.sqlite',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',type=int,nargs='+',help='Years to extract; with this option both classes are selected by default')
    parser.add_argument('--labels',nargs='+',choices=['benign','malware'],help='Classes to extract; requires --years')
    parser.add_argument('--dataset',default='/home/lb/data/Dataset/apks_period')
    parser.add_argument('--output',default='/home/lb/data/Dataset/ForeDroid_output/2014-2017')
    parser.add_argument('--gui-workers',type=int,default=32)
    parser.add_argument('--chain-workers',type=int,default=16)
    parser.add_argument('--export-workers',type=int,default=32,help='Global concurrent APK GUI/OCR exports across all batches')
    parser.add_argument('--min-free-memory-gb',type=float,default=96)
    parser.add_argument('--set-workers',type=int,nargs=2,metavar=('GUI','CALLCHAIN'),help='Update the running controller without restarting it')
    parser.add_argument('--gui-timeout',type=int,default=1800)
    parser.add_argument('--chain-timeout',type=int,default=1800)
    parser.add_argument('--python',default='python3')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--pilot',help='Validate one APK by unique SHA256 prefix')
    args=parser.parse_args()
    if args.labels and not args.years:parser.error('--labels requires --years')
    if Path(args.output).name=='2014-2017' and args.labels and 'malware' in args.labels:parser.error('Training output accepts benign only')
    if args.years and any(y<2000 or y>2100 for y in args.years):parser.error('Invalid year')
    if Path(args.output).name=='2014-2017' and args.years and any(y not in range(2014,2018) for y in args.years):parser.error('Training output accepts 2014–2017 only')
    args.batches = [(y,l) for y in sorted(set(args.years)) for l in dict.fromkeys(args.labels or (['benign'] if y<2018 else ['benign','malware']))] if args.years else list(BATCHES)
    limits=args.set_workers or (args.gui_workers,args.chain_workers)
    if any(n<1 or n>POOL_LIMIT for n in limits):parser.error(f'Worker counts must be between 1 and {POOL_LIMIT}')
    if not 1<=args.export_workers<=64:parser.error('export-workers must be between 1 and 64')
    if args.min_free_memory_gb<0:parser.error('min-free-memory-gb must be nonnegative')
    if args.set_workers:
        database=Path(args.output).expanduser().resolve()/'state.sqlite'
        if not database.is_file():parser.error('No existing batch state database')
        with sqlite3.connect(database.as_uri()+'?mode=rw',uri=True) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_config'").fetchone():
                parser.error('The running controller must first be upgraded to support live concurrency')
            db.executemany('INSERT OR REPLACE INTO runtime_config VALUES (?,?)',zip(('gui_workers','chain_workers'),limits))
        print(f'Concurrency requested: GUI={limits[0]}, callChain={limits[1]}; applies within 30 seconds')
        return
    csv.field_size_limit(2**31-1)
    Runner(args).run()

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from output_layout import dispatch,TRAIN
    dispatch(__file__,'--output',TRAIN,year_flag='--years',training=False)
    main()
