#!/usr/bin/env python3
"""F-Droid provenance adapters; original worker sources/checkpoints stay unchanged."""
from pathlib import Path
import argparse,csv,concurrent.futures as futures,fcntl,hashlib,importlib.util,json,os,shutil,sys,threading,time
from types import SimpleNamespace
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

def bind(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    p=root/'fdroid_adapter.json';value={'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if p.exists() and json.loads(p.read_text())!=value:raise ValueError('F-Droid adapter changed; review this root before resuming')
    if not p.exists():p.write_text(json.dumps(value))

def load(name,folder,file):
    sys.path.insert(0,str(PROJECT/folder))
    spec=importlib.util.spec_from_file_location(name,PROJECT/folder/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def provenance(frame,root,split):
    info=json.loads((Path(root)/'provenance.json').read_text())
    if info['split']!=split:raise ValueError('Unexpected F-Droid experiment split')
    identities=set(map(tuple,info['identities']))
    observed=set(frame[['dataset_year','apk_name','dataset_class']].itertuples(index=False,name=None))
    if not observed<=identities:raise ValueError('Rows not listed in frozen split inventory')
    if split=='train' and (info['dataset']!='F-Droid' or not frame.dataset_year.eq('0').all() or not frame.dataset_class.eq('benign').all()):
        raise ValueError('Non-F-Droid benign data entered training')

def extract(root,view,gui_workers,chain_workers,ocr_workers,pilot=None):
    import batch_extract as b
    bind(root)
    original_call=b.call
    def call(command,cwd,log,timeout=None,env=None):
        if any('apktool_2.9.3.jar' in str(x) for x in command):timeout=900
        return original_call(command,cwd,log,timeout,env)
    b.call=call
    input_gate=threading.Semaphore(8)
    output_gates={name:threading.Semaphore(4) for name in ("gui","callchain")}
    log_gate=threading.Semaphore(2)
    class Runner(b.Runner):
        def staged(self,module,apk,folder,work,log):
            if module=='gui' and b.model_ok(folder/'output'/apk.stem/'appSerialized.txt'):return 'done','existing GUI model'
            if module=='callchain' and b.report_ok(folder/'output'/apk.stem/'CallGraphInfo/rule_analysis_report.json'):return 'done','existing call-chain report'
            local=work/module
            for n in ['output','results','logs']:(local/n).mkdir(parents=True,exist_ok=True)
            local_log=local/'job.log'
            inputs=work/'inputs';inputs.mkdir(exist_ok=True);local_apk=inputs/apk.name
            with input_gate:shutil.copyfile(apk,local_apk)
            try:
                state,detail=(super().gui if module=='gui' else super().chain)(local_apk,local,work,local_log)
                if state in ('done','filtered'):
                    with output_gates[module]:
                        source=local/'output'/apk.stem
                        if source.exists():shutil.copytree(source,folder/'output'/apk.stem,dirs_exist_ok=True)
                        for n in ['results','logs']:
                            for item in (local/n).iterdir():
                                target=folder/n/item.name
                                if item.is_dir():shutil.copytree(item,target,dirs_exist_ok=True)
                                else:shutil.copyfile(item,target)
                return state,detail
            finally:
                if local_log.exists():
                    with log_gate:shutil.copyfile(local_log,log)
        def gui(self,apk,folder,work,log):return self.staged('gui',apk,folder,work,log)
        def chain(self,apk,folder,work,log):return self.staged('callchain',apk,folder,work,log)
    args=SimpleNamespace(output=str(root),dataset=str(view),batches=[(0,'benign')],gui_workers=gui_workers,chain_workers=chain_workers,
        export_workers=ocr_workers,min_free_memory_gb=128,gui_timeout=1800,chain_timeout=1800,python=sys.executable,retry_failed=False,pilot=pilot)
    csv.field_size_limit(sys.maxsize);Runner(args).run()

def scenario():
    m=load('fdroid_scenarios','cluster_pipeline','scenario_worker.py')
    def prepare(self):
        frame=m.pd.read_csv(self.source,dtype=str,keep_default_na=False)
        required={'id','apk_name','generated_description','label','context','rawtext','entry_point','entry_method','entry_point_actions','dataset_year','dataset_class'}
        if required-set(frame):raise ValueError('Missing scenario columns')
        if not frame.id.is_unique or frame.generated_description.str.strip().eq('').any():raise ValueError('Invalid IDs/descriptions')
        provenance(frame,self.a.root,self.a.split)
        mask=m.gui_mask(frame);frame=frame[mask if self.branch=='gui' else ~mask].copy().reset_index(drop=True)
        m.atomic_pickle(self.cache/'input.pkl',frame);self.update(len(frame),len(frame),'F-Droid split provenance validated')
    original=m.Worker.__init__
    def init(self,args):bind(args.root);original(self,args)
    m.Worker.__init__=init;m.Worker.prepare=prepare;m.main()

def anomaly():
    m=load('fdroid_anomaly','anomaly_detection_pipeline','worker.py')
    def metadata(self,split):
        frame=m.pd.read_csv(self.source(split),sep=';',usecols=['id','apk_name','cluster','dataset_year','dataset_class','generated_description'],dtype=str,keep_default_na=False)
        if not frame.id.is_unique or not frame.generated_description.str.strip().ne('').all():raise ValueError('Invalid IDs/descriptions')
        provenance(frame,(self.a.training_root or self.root) if split=='train' else self.root,split)
        frame['cluster']=m.pd.to_numeric(frame.cluster,errors='raise').astype(int).astype(str);return frame
    original=m.Worker.__init__
    def init(self,args):
        if args.stage not in ('embedding','references','scores'):raise ValueError('Use coverage-normalized mean trainer for OCSVM')
        bind(args.root);original(self,args)
    m.Worker.__init__=init;m.Worker.metadata=metadata;m.main()

if __name__=='__main__':
    kind=sys.argv.pop(1)
    if kind=='extract':
        p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
        p.add_argument('--pilot');p.add_argument('--gui-workers',type=int,default=32);p.add_argument('--chain-workers',type=int,default=32);p.add_argument('--ocr-workers',type=int,default=16)
        a=p.parse_args();extract(a.root,a.view,a.gui_workers,a.chain_workers,a.ocr_workers,a.pilot)
    elif kind=='scenario':scenario()
    elif kind=='anomaly':anomaly()
    else:raise ValueError('Unknown worker')
