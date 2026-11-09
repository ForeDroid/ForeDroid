#!/usr/bin/env python3
"""Evaluate saved RQ2 assertions with auditable static identity evidence.
Does not claim call-chain localization accuracy or official hash/family mapping.
"""
from pathlib import Path
import argparse,json,csv,hashlib,html,collections,re

def metrics(tp,fp,fn):
 p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
 return dict(TP=tp,FP=fp,FN=fn,precision=p,recall=r,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
def write_csv(path,rows):
 if rows:
  with path.open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/lb/data/Dataset/ForeDroid_output/RQ2/GPMalware'));a=p.parse_args();out=a.root/'evaluation'
 inputs=json.loads((out/'evaluation_inputs.json').read_text());candidates=json.loads((out/'identity_candidates_UNVERIFIED.json').read_text())
 predictions={}
 for apk,record in inputs['reports'].items():
  assertions=[]
  for line in record['matched_section'].splitlines():
   match=re.match(r'^\s*-\s+\*\*(T(?:[1-9]|1\d|2[0-5])):\s*[^*]+\*\*(.*)$',line)
   if not match:continue
   suffix=match.group(2)
   # Count the report's declared matches, including qualified/weak matches.
   # Exclude explicit negative examples, rather than re-judging their evidence.
   if re.match(r'\s+is\s+(?:\*\*)?(?:not supported|not matched|unsupported|tempting)',suffix,re.I):continue
   assertions.append(dict(label=match.group(1),quote=line.strip()))
  predictions[apk]=assertions
 identities=[];claims=[];annotations=[];app=[];labelrows=[]
 unresolved={'AdClickerBN','ToastAmigo'}
 for c in candidates:
  family=c['family'];best,second=c['top'][:2];apk=best['apk'] if family not in unresolved else ''
  strong=bool(apk and best['score']>=15 and best['score']-second['score']>=8)
  identity_status='strong_static_evidence' if strong else ('candidate_needs_manual_confirmation' if apk else 'unresolved')
  identities.append(dict(family=family,apk=apk,identity_status=identity_status,anchors='; '.join(best['anchors']),score=best['score'],runner_up_score=second['score']))
  gtrows=inputs['ground_truth'][family];truth={t for row in gtrows for t in row['labels']}
  pred={v['label'] for v in predictions.get(apk,[])}
  status='report_available' if apk in predictions else ('identity_unresolved' if not apk else ('report_failed' if (a.root/'reports/0'/f'{apk}_failed.txt').exists() else 'no_report'))
  tp,fp,fn=truth&pred,pred-truth,truth-pred
  m=metrics(len(tp),len(fp),len(fn)) if apk else dict.fromkeys(['TP','FP','FN','precision','recall','f1'],'')
  app.append(dict(family=family,apk=apk,identity_status=identity_status,report_status=status,gt_labels=' '.join(sorted(truth,key=lambda t:int(t[1:]))),predicted_labels=' '.join(sorted(pred,key=lambda t:int(t[1:]))),gt_unique_count=len(truth),report_unique_count=len(pred) if apk else '',tp_labels=' '.join(sorted(tp,key=lambda t:int(t[1:]))) if apk else '',fp_labels=' '.join(sorted(fp,key=lambda t:int(t[1:]))) if apk else '',fn_labels=' '.join(sorted(fn,key=lambda t:int(t[1:]))) if apk else '',**m))
  for t in [f'T{i}' for i in range(1,26)]:
   result='UNRESOLVED_IDENTITY' if not apk else ('TP' if t in tp else 'FP' if t in fp else 'FN' if t in fn else 'TN')
   evidence=[v['quote'] for v in predictions.get(apk,[]) if v['label']==t]
   labelrows.append(dict(family=family,apk=apk,identity_status=identity_status,label=t,ground_truth=int(t in truth),predicted=int(t in pred) if apk else '',result=result,prediction_evidence='\n'.join(evidence)))
  for idx,row in enumerate(gtrows,1):
   covered=set(row['labels'])&pred
   row_status='UNRESOLVED_IDENTITY' if not apk else ('ALL_TYPES_PRESENT' if covered==set(row['labels']) else 'SOME_TYPES_PRESENT' if covered else 'NO_TYPES_PRESENT')
   annotations.append(dict(family=family,apk=apk,identity_status=identity_status,annotation_row=idx,source_line=row['source_line'],call_chain=row['chain'],description=row['description'],gt_labels=' '.join(row['labels']),types_found=' '.join(sorted(covered)),type_coverage_status=row_status,call_chain_localization='NOT_EVALUATED: type coverage does not prove this path was found'))
  for v in predictions.get(apk,[]):claims.append(dict(family=family,apk=apk,label=v['label'],result='TYPE_TP' if v['label'] in truth else 'TYPE_FP',quote=v['quote'],identity_status=identity_status))
 summary={}
 for scope in ['strong_static_evidence','extended_candidates']:
  selected=[r for r in app if r['apk'] and (scope=='extended_candidates' or r['identity_status']=='strong_static_evidence')]
  summary[scope]=dict(apks=len(selected),reports=sum(r['report_status']=='report_available' for r in selected),**metrics(*(sum(r[k] for r in selected) for k in ['TP','FP','FN'])))
  summary[scope]['per_type']={t:metrics(*(sum(r['result']==k for r in labelrows if r['label']==t and any(v['family']==r['family'] for v in selected)) for k in ['TP','FP','FN'])) for t in [f'T{i}' for i in range(1,26)]}
 summary.update(total_ground_truth_files=len(app),ground_truth_chain_rows=len(annotations),parsed_reports=len(predictions),unresolved_families=sorted(unresolved),unit='one unique APK × taxonomy label, deduplicated',limitations=['Identity is inferred from static evidence, not an official hash mapping. Strong and candidate scopes are separate.','Unresolved identities are not silently treated as negatives.','Mapped APKs without reports have empty predictions; all annotated types count as FN.','Per-annotation rows measure TYPE COVERAGE ONLY, not call-chain localization.','Predictions are deduplicated IDs declared in report Matched Behavior Types bullets, excluding explicit negative examples; no model re-judgment.','No ground truth was used to change predictions or report-generation parameters.'])
 write_csv(out/'apk_metrics.csv',app);write_csv(out/'label_decisions.csv',labelrows);write_csv(out/'annotation_row_coverage.csv',annotations);write_csv(out/'prediction_claims.csv',claims);write_csv(out/'identity_mapping.csv',identities)
 (out/'metrics.json').write_text(json.dumps(summary,indent=2))
 def esc(s):return html.escape(str(s))
 sections=[]
 for scope in ['strong_static_evidence','extended_candidates']:
  m=summary[scope];sections.append(f'<section><h2>{esc(scope)}</h2><p>{m["apks"]} APK · {m["reports"]} reports · TP {m["TP"]} / FP {m["FP"]} / FN {m["FN"]}</p><div class="scores">Precision {m["precision"]:.2%} &nbsp; Recall {m["recall"]:.2%} &nbsp; F1 {m["f1"]:.2%}</div></section>')
 details=[]
 for row in app:
  family=row['family'];records=[v for v in labelrows if v['family']==family and v['result'] not in ['TN'] and (v['ground_truth'] or v['predicted'])]
  chains=[v for v in annotations if v['family']==family]
  details.append('<details><summary>'+esc(family)+' · '+esc(row['identity_status'])+' · TP '+esc(row['TP'])+' / FP '+esc(row['FP'])+' / FN '+esc(row['FN'])+'</summary><p>APK: '+esc(row['apk'] or 'UNRESOLVED')+' · '+esc(row['report_status'])+'</p><table><tr><th>Type</th><th>Result</th><th>Report quote</th></tr>'+''.join('<tr><td>'+esc(v['label'])+'</td><td class="'+v['result']+'">'+esc(v['result'])+'</td><td>'+esc(v['prediction_evidence'])+'</td></tr>' for v in records)+'</table><h4>Ground-truth rows: type coverage, not path localization</h4>'+''.join('<p><b>#'+str(v['annotation_row'])+' '+esc(v['gt_labels'])+' · '+esc(v['type_coverage_status'])+'</b><br>'+esc(v['call_chain'])+'</p>' for v in chains)+'</details>')
 page='''<!doctype html><meta charset="utf-8"><title>ForeDroid RQ2 evaluation</title><style>body{background:#0b1220;color:#dde7f4;font:15px system-ui;max-width:1150px;margin:36px auto;padding:20px}h1,h2{color:#92d8ff}section,details{background:#142137;padding:20px;margin:16px 0;border-radius:12px}.scores{font-size:24px;color:#77e2bf}summary{cursor:pointer;font-weight:bold}table{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #34445b;padding:10px}.FP,.FN{color:#ffae91}.TP{color:#77e2bf}input{padding:12px;width:95%;background:#142137;color:white;border:1px solid #678;border-radius:8px}p{overflow-wrap:anywhere}</style><h1>ForeDroid · RQ2 evaluation</h1><p>统计单位：APK × T类别。身份推断存在不确定性；以下分开报告。逐条类别覆盖不等于调用链定位正确。</p>'''+''.join(sections)+'<h2>Limitations</h2><ul>'+''.join('<li>'+esc(s)+'</li>' for s in summary['limitations'])+'</ul><p>Unresolved: '+', '.join(sorted(unresolved))+'</p><h2>Per-APK and per-annotation decisions</h2><input placeholder="Search family / APK hash" oninput="document.querySelectorAll(\'details\').forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(this.value.toLowerCase()))">'+''.join(details)
 (out/'evaluation_report.html').write_text(page)
 manifest={str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'evaluation_inputs.json',out/'identity_candidates_UNVERIFIED.json']}
 (out/'input_sha256.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_type'} if isinstance(v,dict) else v for k,v in summary.items()},indent=2))
if __name__=='__main__':main()
