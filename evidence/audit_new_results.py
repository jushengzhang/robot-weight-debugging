from pathlib import Path
import json, math, hashlib
ROOT=Path(__file__).resolve().parent
R=ROOT/'raw/remote186'
W=ROOT/'raw/fastwam'
OUT=ROOT

def read(p):
 p=Path(p)
 return json.loads(p.read_text()) if p.suffix=='.json' else [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def key(r):return (r['suite'],int(r.get('variant_id',r.get('id',0))),int(r.get('seed',7)),int(r.get('init_idx',0)))
def paired(o,e,select=lambda r:True,pro=False,exclude_errors=False):
 kd=(lambda r:r['key']) if pro else key
 od={kd(r):r for r in o};ed={kd(r):r for r in e}
 assert len(od)==len(o) and len(ed)==len(e)
 keys=sorted(set(od)&set(ed))
 keys=[k for k in keys if select(od[k]) and (not exclude_errors or (od[k].get('termination')!='error' and ed[k].get('termination')!='error'))]
 n=len(keys);a=sum(bool(od[k]['success']) for k in keys);b=sum(bool(ed[k]['success']) for k in keys)
 g=sum(not od[k]['success'] and ed[k]['success'] for k in keys);l=sum(od[k]['success'] and not ed[k]['success'] for k in keys)
 m=g+l;p=1.0 if not m else min(1.0,2*sum(math.comb(m,i) for i in range(min(g,l)+1))/2**m)
 delta=(g-l)/n;se=math.sqrt((m/n-delta**2)/n)
 return dict(n=n,original=a,edited=b,original_pct=100*a/n,edited_pct=100*b/n,gain=g,loss=l,delta_pp=100*delta,p=p,ci95_pp=[100*(delta-1.96*se),100*(delta+1.96*se)],original_errors=sum(od[k].get('termination')=='error' for k in keys),edited_errors=sum(ed[k].get('termination')=='error' for k in keys))

audit=json.loads((ROOT/'raw/embedding-audit/embedding-audit.json').read_text())
o=read(R/'heldout/lang_all/original_results.json')
result={'pi05_language':{},'controls':{},'suffix':{},'protocol':{'heldout_seed':7,'language_n':765,'editing_sessions_independent':True,'same_heldout_population_across_sessions':True}}
orig_safety=read(R/'heldout/pi_lang_a/original_results.json')
for short in ('a','b','c'):
 e=read(R/f'heldout/lang_all/pi_lang_{short}/results.jsonl')
 hits={tuple((r['suite'],r['variant_id'],r['seed'])):bool(r['edited_token_ids']) for r in audit[short]['heldout_hits']}
 entry=paired(o,e)
 entry['rows']=audit[short]['n_rows'];entry['coefficients_changed']=audit[short]['coefficients_changed'];entry['fraction_parameters_percent']=audit[short]['fraction_parameters_percent']
 entry['touched']=paired(o,e,lambda r:hits[(r['suite'],r['variant_id'],r['seed'])])
 entry['untouched']=paired(o,e,lambda r:not hits[(r['suite'],r['variant_id'],r['seed'])])
 assert entry['untouched']['gain']==entry['untouched']['loss']==0
 entry['difficulty']={str(d):paired(o,e,lambda r:r['difficulty']==d) for d in range(1,6)}
 safety=read(R/f'heldout/pi_lang_{short}/pi_lang_{short}/results.jsonl')
 entry['other_categories']=paired(orig_safety,safety,lambda r:r['category']!='Language Instructions')
 entry['initial_200']=paired(orig_safety,safety,lambda r:r['category']=='Language Instructions')
 result['pi05_language'][short]=entry
controls={}
for name in ('ctrl_dict_full','ctrl_random_dir'):
 controls[name]=read(R/f'heldout/lang_ctrl/{name}/results.jsonl')
 result['controls'][name]=paired(o,controls[name])
 result['controls'][name+'_valid_pairs']=paired(o,controls[name],exclude_errors=True)
result['controls']['editor_vs_dictionary']=paired(controls['ctrl_dict_full'],read(R/'heldout/lang_all/pi_lang_c/results.jsonl'),exclude_errors=True)
for model,base in [('FastWAM',W/'runs'),('pi05',R/'runs')]:
 tagged=read(base/'baseline210_raw/results.jsonl') if model=='pi05' else read(base/'baseline210/results.jsonl')
 clean=read(base/'baseline210_strip/results.jsonl')
 result['suffix'][model]=paired(tagged,clean)
 result['suffix'][model]['categories']={c:paired(tagged,clean,lambda r:r['category']==c) for c in sorted({r['category'] for r in tagged})}

pro_o=read(R/'heldout_pro/pro_a_big/original_results.json')
pro_e=read(R/'heldout_pro/pro_a_big/pro_a/results.jsonl')
result['pro_archived']={}
for cat in ('ALL perturbed','object','position','semantic','task','environment','original'):
 result['pro_archived'][cat]=paired(pro_o,pro_e,lambda r:(r.get('ptype')!='original' if cat=='ALL perturbed' else r.get('ptype')==cat),pro=True,exclude_errors=True)

assert result['pi05_language']['c']['original']==660 and result['pi05_language']['c']['edited']==704
assert result['pi05_language']['c']['touched']['n']==335 and result['pi05_language']['c']['untouched']['n']==430
assert result['pi05_language']['c']['touched']['gain']==48 and result['pi05_language']['c']['touched']['loss']==4
assert result['controls']['editor_vs_dictionary']['gain']==20 and result['controls']['editor_vs_dictionary']['loss']==9
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'new-results-verified.json').write_text(json.dumps(result,indent=2)+'\n')
summary={k:{'n':v['n'],'original':v['original'],'edited':v['edited'],'g/l':[v['gain'],v['loss']],'p':v['p'],'untouched_n':v['untouched']['n'],'other_g/l':[v['other_categories']['gain'],v['other_categories']['loss']]} for k,v in result['pi05_language'].items()}
print(json.dumps(summary,indent=2))
print('controls',json.dumps(result['controls']))
print('pro',json.dumps(result['pro_archived'].get('ALL perturbed')))
