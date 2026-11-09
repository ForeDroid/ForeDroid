"""Training/test output routing with a single unified testing run."""
from pathlib import Path
import subprocess
import sys

BASE=Path('/home/lb/data/Dataset/ForeDroid_output')
TRAIN=BASE/'2014-2017'
TEST=BASE/'2018-2022'
PROJECT=Path(__file__).resolve().parent
def groups(root,years=None):
    root=Path(root)
    if root.name!='2018-2022':return [(root,years)]
    selected=sorted(set(years or range(2018,2023)))
    if not set(selected)<=set(range(2018,2023)):raise ValueError('Testing root accepts years 2018–2022 only')
    return [(root/'runs',selected)]


def dispatch(script,root_flag,default,year_flag=None,training=False,monitor=False):
    """Map the testing collection to its single computation root."""
    argv=sys.argv[1:]
    if '--help' in argv or '-h' in argv:return
    def values(flag):
        if flag not in argv:return []
        i=argv.index(flag)+1;j=i
        while j<len(argv) and not argv[j].startswith('--'):j+=1
        return argv[i:j]
    roots=values(root_flag);root=Path(roots[0] if roots else default)
    if root.name!='2018-2022':return
    if monitor:
        args=[sys.executable,str(PROJECT/'monitor_outputs.py'),'--output',str(root)]
        if '--once' in argv:args+=['--once']
        raise SystemExit(subprocess.call(args))
    years=list(map(int,values(year_flag))) if year_flag and values(year_flag) else None
    selected=groups(root,years)
    def remove(flag):
        if flag not in argv:return
        i=argv.index(flag);j=i+1
        while j<len(argv) and not argv[j].startswith('--'):j+=1
        del argv[i:j]
    remove(root_flag)
    if year_flag:remove(year_flag)
    for folder,ys in selected:
        command=[sys.executable,str(script),*argv,root_flag,str(folder)]
        if year_flag:command += [year_flag,*map(str,ys)]
        if training and '--training-root' not in argv:command+=['--training-root',str(TRAIN)]
        if Path(script).name=='generate_description.py' and '--splits' not in argv:command+=['--splits','test']
        print('OUTPUT ROOT:',folder,flush=True)
        subprocess.run(command,check=True)
    if Path(script).name=='run_inference.py':
        subprocess.run([sys.executable,str(PROJECT/'anomaly_detection_pipeline/summarize_testing.py'),'--root',str(root)],check=True)
    raise SystemExit(0)

def replace_if_changed(pending,target):
    """Publish deterministic CSVs without invalidating checkpoints for identical data."""
    pending=Path(pending);target=Path(target)
    if target.exists() and target.stat().st_size==pending.stat().st_size:
        with target.open('rb') as a,pending.open('rb') as b:
            while True:
                x=a.read(1024*1024);y=b.read(1024*1024)
                if x!=y:break
                if not x:
                    pending.unlink();return
    pending.replace(target)


def raw_root(root):
    root=Path(root)
    return root.parent if root.name=='runs' and root.parent.name=='2018-2022' else root
