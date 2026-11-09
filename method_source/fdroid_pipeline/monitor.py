#!/usr/bin/env python3
"""Read-only F-Droid reference-library dashboard, matching download monitors."""
import argparse,curses,json,locale,os,shutil,sqlite3,subprocess,sys,time,unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path('/home/lb/data/Dataset/ForeDroid_output/F-Droid')
NAMES={'extract':'Backstage / ICCBot / OCR 提取','gui_merge':'GUI merge','fusion':'GUI与调用链融合','prepare_training':'整理全部F-Droid训练行为','train_descriptions':'训练集M3描述','train_scenarios':'训练集场景聚类','train_merge_scenarios':'训练场景合并','train_embedding':'训练行为嵌入','train_references':'构建良性参考库','train_scores':'训练行为异常评分','fit_ocsvm':'训练改进单模型OCSVM','training_coverage':'统计训练参考库实际覆盖率','prepare_historical':'准备2018–2022测试','finished':'全部完成','failed':'任务停止，请查看日志'}

def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
def width(s):return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in str(s))
def fit(s,n):
    out=''
    for c in str(s):
        if width(out+c)>n:break
        out+=c
    return out+' '*max(0,n-width(out))
def bar(percent,n=20):
    filled=int(max(0,min(100,percent))*n/100);return '█'*filled+'░'*(n-filled)
def duration(seconds):
    if seconds is None:return '待估算'
    seconds=max(0,int(seconds));days,seconds=divmod(seconds,86400);hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    if days:return f'{days}天{hours:02}时{minutes:02}分'
    if hours:return f'{hours}时{minutes:02}分'
    return f'{minutes}分{seconds:02}秒'
def extraction_times(root,e,rows,now):
    start=e.get('started_at');elapsed=max(0,now-start) if start else 0
    counts={}
    try:
        path=(root/'extraction/state.sqlite').resolve()
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2) as db:
            since=max(start or now,now-600)
            counts=dict(db.execute('SELECT module,count(*) FROM events WHERE finished_at>=? GROUP BY module',(since,)))
    except (OSError,ValueError,sqlite3.Error):pass
    for r,module in zip(rows,['gui','callchain']):
        remaining=r['total']-r['closed'];n=counts.get(module,0)
        if remaining==0:r['eta']=0
        elif elapsed>=60 and n>=5:r['eta']=remaining/(n/min(600,elapsed))
        elif elapsed>=60 and r['closed']>=5:r['eta']=remaining/(r['closed']/elapsed)
        else:r['eta']=None
    if not rows:return None
    return max(r['eta'] for r in rows) if all(r['eta'] is not None for r in rows) else None
def snapshot(root):
    s=read(root/'status.json');e=read(root/'extraction/status.json')
    rows=[]
    for module,title in [('gui','Backstage'),('callchain','ICCBot')]:
        counts={r['status']:r['count'] for r in e.get('counts',[]) if r['module']==module}
        total=sum(counts.values()) or s.get('total_apks',4933)
        closed=sum(v for k,v in counts.items() if k not in ('pending','running'))
        rows.append(dict(title=title,total=total,closed=closed,done=counts.get('done',0),running=counts.get('running',0),failed=counts.get('failed',0),timeout=counts.get('timeout',0),filtered=counts.get('filtered',0)))
    descriptions=[('训练集M3',read(root/'training/description_status.json'))]
    return s,e,rows,descriptions

def lines(root):
    s,e,rows,descriptions=snapshot(root);done=len(s.get('completed_stages',[]));total=s.get('stage_count',18);stage=s.get('stage','等待启动');now=time.time()
    starts=[v for v in [s.get('started_at'),e.get('started_at')] if isinstance(v,(int,float))]
    end=s.get('updated_at',now) if s.get('state') in ('finished','failed') else now
    elapsed=max(0,end-min(starts)) if starts else None
    eta=extraction_times(root,e,rows,now)
    finish=datetime.fromtimestamp(now+eta,ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M') if eta is not None else '待估算'
    updated=e.get('updated_at') if stage=='extract' else s.get('updated_at')
    age=f'{max(0,int(now-updated))}秒前' if updated else '等待更新'
    text=[f" F-Droid 参考库实验   {datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S')} 北京时间",'',
          f" 总流程  {bar(done/total*100)} {done}/{total} 阶段   {s.get('state','等待')}",
          f" 当前：{NAMES.get(stage,stage)}",f" APK {s.get('total_apks',4933):,} · 训练 {s.get('training_apks',4933):,} · 测试：2018–2022",
          f" 已运行 {duration(elapsed)}   最近状态更新 {age}"]
    if stage=='train_descriptions':
        d=descriptions[0][1]; count=read(root/'fusion_summary.json').get('training_rows',0)
        completed=d.get('rows_done',0);remaining=max(0,count-completed)
        spent=max(1,now-d.get('started_at',now));rate=completed/spent
        left=remaining/rate if rate else None
        percent=100*completed/count if count else 0
        when=datetime.fromtimestamp(now+left,ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M') if left is not None else '待估算'
        text+=['', ' ━━━ 当前阶段：M3 描述生成 ━━━',
               f" {bar(percent,28)}  {percent:.2f}%",
               f" 已写入 {completed:,} / 总计 {count:,} 条   剩余 {remaining:,} 条",
               f" 平均速度 {rate*60:.1f} 条/分钟   本阶段已运行 {duration(spent)}",
               f" 本阶段预计剩余 {duration(left)}   预计完成 {when}（北京时间）",
               f" API成功 {d.get('api_done',0):,}   缓存复用 {d.get('reused_rows',0):,}   排队/处理中 {d.get('inflight_or_queued',0):,}",
               f" 并发 {d.get('workers','—')}   正在重试 {d.get('retrying',0)}   累计请求错误 {d.get('request_errors',0)}",
               f" 描述状态更新：{max(0,int(now-d.get('updated_at',now)))}秒前",
               ' 逐条写入并持续补充请求；恢复初期复用缓存，速度与估时需等待稳定。',
               ' 后续：场景聚类 → 向量嵌入 → 参考库 → 训练评分 → OCSVM训练 → 测试评估', '']
    if stage=='extract':
        text+=[f" 提取预计剩余 {duration(eta)}   预计完成 {finish}（北京时间）",
               ' '+ '   '.join(f"{r['title']}剩余约 {duration(r['eta'])}" for r in rows)]
    elif stage!='train_descriptions':text+=[' 提取阶段已结束；完整流程剩余时间待估算。']
    text+=['',' 模块       '+fit('处理进度',25)+fit('成功',9)+fit('运行',8)+fit('失败',8)+fit('超时',8)+'过滤']
    for r in rows:
        p=r['closed']/r['total']*100 if r['total'] else 0
        text.append(' '+fit(r['title'],11)+fit(f"{bar(p,12)} {r['closed']}/{r['total']}",25)+fit(r['done'],9)+fit(r['running'],8)+fit(r['failed'],8)+fit(r['timeout'],8)+str(r['filtered']))
    ex=e.get('export_progress',{})
    text+=['',f" OCR/GUI导出  成功 {ex.get('done',0)}   失败 {ex.get('failed',0)}   剩余 {ex.get('remaining','—')}"]
    for title,d in descriptions:
        text.append(f" {title}  写入 {d.get('rows_done',0):,}   API成功 {d.get('api_done',0):,}   重试中 {d.get('retrying',0)}   {d.get('state','等待')}")
    text+=['',' 时间按处理速度估算，失败/过滤也计入已处理；不含OCR及后续阶段。',
           ' 完整流程剩余：待估算（尚无本批后续阶段耗时）。',
           ' 训练参考库：training/   测试结果：testing/2018-2022/']
    if s.get('error'):text+=[' 错误：'+s['error']]
    text+=[' 日志：'+str(root/'controller.log')]
    return text

def dashboard(screen,root,interval):
    try:curses.curs_set(0)
    except curses.error:pass
    if curses.has_colors():
        curses.start_color();curses.use_default_colors()
        for i,c in enumerate([curses.COLOR_CYAN,curses.COLOR_GREEN,curses.COLOR_YELLOW,curses.COLOR_RED],1):curses.init_pair(i,c,-1)
    screen.timeout(200);due=0;text=[]
    while True:
        if time.monotonic()>=due:text=lines(root);due=time.monotonic()+interval
        screen.erase();height,columns=screen.getmaxyx()
        def put(y,s,color=0):
            if 0<=y<height-1:
                try:screen.addstr(y,0,fit(s,max(0,columns-1)),curses.color_pair(color))
                except curses.error:pass
        for y,s in enumerate(text):
            if y>=height-3:break
            put(y,s,1 if y==0 else 4 if s.startswith(' 错误') else 0)
        put(height-3,f' 每{interval:g}秒刷新；提取成功不等于最终可评分，失败/过滤/超时单独保留。')
        put(height-2,' q 退出监控 · 1 查看运行终端 · Ctrl+B再D 返回监控');screen.refresh()
        key=screen.getch()
        if key in (ord('q'),ord('Q'),27):return False
        if key==ord('1') and shutil.which('tmux'):
            if os.environ.get('TMUX'):subprocess.run(['tmux','select-window','-t','foredroid_fdroid:pipeline'])
            else:return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=float,default=2);a=p.parse_args()
    if a.interval<1:p.error('interval must be >=1')
    locale.setlocale(locale.LC_ALL,'')
    if a.once or not sys.stdout.isatty():print('\n'.join(lines(a.root)));return
    while curses.wrapper(dashboard,a.root,a.interval):
        subprocess.run(['tmux','select-window','-t','foredroid_fdroid:pipeline']);subprocess.run(['tmux','attach-session','-t','foredroid_fdroid'])
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
    except curses.error:print('请在SSH交互终端运行，或使用 --once。')
