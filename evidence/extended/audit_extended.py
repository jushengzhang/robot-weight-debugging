from pathlib import Path
import json,math,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE/'additional-evidence'
def key(x):return x['suite'],x['variant_id'],x['seed'],x.get('init_idx',0)
def pair(rows):
 d={}
 for x in rows:
  if x['category']!='Language Instructions':continue
  d[key(x),x['arm']]=x
 keys=sorted({k for k,a in d});out=[]
 for k in keys:
  if (k,'orig') in d and (k,'edit') in d:out.append((d[k,'orig'],d[k,'edit']))
 return out
rows=[]
for i in [18,19,20,21,33,34,35,36,37,48,49,50]:rows.extend(json.loads(x) for x in (R/f'validation/roll_{i}.jsonl').read_text().splitlines())
raw=pair(rows)
# Same-seed infrastructure retries in request 24 replace the affected records, not new trials.
rows.extend(json.loads(x) for x in (R/'validation/roll_24.jsonl').read_text().splitlines())
reconciled=pair(rows)
def stats(pairs):
 n=len(pairs);a=sum(x['success'] for x,y in pairs);b=sum(y['success'] for x,y in pairs);g=sum(not x['success'] and y['success'] for x,y in pairs);l=sum(x['success'] and not y['success'] for x,y in pairs);m=g+l
 p=min(1,2*sum(math.comb(m,i) for i in range(min(g,l)+1))/2**m) if m else 1
 return dict(n=n,original=a,edited=b,gain=g,loss=l,delta_pp=100*(b-a)/n,exact_p=p,original_errors=sum(x.get('termination')=='error' for x,y in pairs),edited_errors=sum(y.get('termination')=='error' for x,y in pairs))
assert len({(x['suite'],x['variant_id']) for x,y in reconciled})==577
result={'practice_validation_raw':stats(raw),'practice_validation_reconciled':stats(reconciled),'practice_by_seed':{str(s):stats([(x,y) for x,y in reconciled if x['seed']==s]) for s in sorted({x['seed'] for x,y in reconciled})}}
assert (result['practice_validation_reconciled']['n'],result['practice_validation_reconciled']['original'],result['practice_validation_reconciled']['edited'])==(1533,1356,1376)
m=json.load(open(R/'probe-manifest.json'));probes={i:{(x['idx'],x['arm']):np.load(R/x['local_action']) for x in rows} for i,rows in m['probes'].items()}
affected=[i for i in range(16) if not np.array_equal(probes['23'][i,'orig'],probes['23'][i,'edit'])]
final=np.concatenate([(probes['23'][i,'edit']-probes['23'][i,'orig']).ravel() for i in affected]);controls={}
for label,k in [('norm_only','38'),('direction_only','39')]:
 delta=np.concatenate([(probes[k][i,'edit']-probes['23'][i,'orig']).ravel() for i in affected]);controls[label]={'relative_displacement_norm':float(np.linalg.norm(delta)/np.linalg.norm(final)),'cosine_with_final_displacement':float(np.dot(delta,final)/(np.linalg.norm(delta)*np.linalg.norm(final))),'unchanged_controls':sum(np.array_equal(probes[k][i,'edit'],probes[k][i,'orig']) for i in [3,5,8,10,11])}
result['replay_controls']=controls;result['replay_inputs']=[]
for i,j in [(0,3),(1,3),(2,3),(4,5),(6,8),(7,8),(9,10)]:
 target=probes['23'][j,'orig'];dist={label:float(np.linalg.norm(probes[k][i,arm]-target)) for label,k,arm in [('original','23','orig'),('norm_only','38','edit'),('direction_only','39','edit'),('final','23','edit')]}
 result['replay_inputs'].append({'input_id':i,'prompt':m['prompts'][str(i)],'anchor_prompt':m['prompts'][str(j)],'distances':dist})
result['median_relative_distance']={label:float(np.median([x['distances'][label]/x['distances']['original'] for x in result['replay_inputs']])) for label in ['original','norm_only','direction_only','final']}
result['original_replay_exact_count']=sum(np.array_equal(probes['23'][i,'orig'],probes['39'][i,'orig']) for i in range(16))
result['source_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(HERE/'extended-evidence-verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','replay_inputs']},indent=2))
print('Prompts',[x['prompt'] for x in result['replay_inputs']])
