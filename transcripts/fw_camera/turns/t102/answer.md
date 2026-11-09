The stronger attention ablation also fails to establish a benefit: +1.6 points with a 2.9-point standard error. I’ll revert it and stop adding hypotheses. The remaining passes will give the original frozen readout candidate a higher-powered final assessment, including all six non-camera categories. If it still lacks a clear independent gain or shows unacceptable damage, I’ll retain the original model.
```python
m27=paired_summary(paired_rows([27]))
ledger=json.loads((root/"pass_ledger.json").read_text())
ledger.extend([
    {"pass":26,"type":"probe","edit":ablation_label,"metrics":summary26},
    {"pass":27,"type":"rollout","edit":ablation_label,"purpose":"Discovery",**m27}
])
(root/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
for name in list(edited()):
    drop_edit(name)
old_dir=root/"candidates"/"readout_camera_rank2_half"
print("Frozen candidate files:",[(p.name,p.stat().st_size) for p in old_dir.rglob("*") if p.is_file()])

# New camera runs are fixed now, before their results:
# every one of the 602 selection-disjoint variants gets two independent seeds;
# a hash-selected 166 get a third seed, yielding 1370 total paired evaluations
# including passes20-23.
old_validation_eps=[e for panel in panels for e in panel]
old_keys=[(e["suite"],e["id"]) for e in old_validation_eps]
reserve_keys=[(e["suite"],e["id"]) for e in final_camera_reserve]
assert len(set(old_keys+reserve_keys))==602
all_eval_keys=old_keys+reserve_keys
extra=[{"suite":suite,"id":vid,"seed":180000+i}
       for i,(suite,vid) in enumerate(all_eval_keys)]
third_keys=sorted(all_eval_keys,key=lambda k:hashlib.sha256(
    f"third-seed-fixed:{k[0]}:{k[1]}".encode()).hexdigest())[:166]
extra += [{"suite":suite,"id":vid,"seed":190000+i} for i,(suite,vid) in enumerate(third_keys)]
extra.sort(key=lambda e:hashlib.sha256(
    f'final-camera-order:{e["suite"]}:{e["id"]}:{e["seed"]}'.encode()).hexdigest())
remaining_camera_panels=[final_camera_reserve]+[extra[i:i+128] for i in range(0,len(extra),128)]
assert [len(p) for p in remaining_camera_panels]==[128]*7
camera_pass_assignment=dict(zip([28,29,30,31,38,39,40],remaining_camera_panels))
safety_categories=["Light Conditions","Robot Initial States","Language Instructions",
                   "Background Textures","Sensor Noise","Objects Layout"]
safety_requests={32+i:{"type":"rollout","model":"both",
                       "sample":{"category":cat,"n":128,"seed":220000+i}}
                 for i,cat in enumerate(safety_categories)}
plan={
    "candidate":"readout_camera_rank2_half",
    "tensor_sha256":json.loads(lock_path.read_text())["tensor_sha256"],
    "no_further_weight_tuning":True,
    "camera_rule":"Retain only with positive independent camera gain exceeding 1.96 times variant-cluster SE",
    "safety_rule":"No noncamera category point-estimate drop greater than 2 percentage points; report uncertainty",
    "fallback":"original",
    "camera_episodes_total_including_pass20_23":1370,
    "camera_unique_selection_disjoint_variants":602,
    "camera_panels":camera_pass_assignment,
    "safety_requests":safety_requests
}
(root/"final_evaluation_plan.json").write_text(json.dumps(plan,indent=2))
for k,eps in camera_pass_assignment.items():
    (root/f"request_{k}.json").write_text(json.dumps({"type":"rollout","model":"both","episodes":eps}))
for k,req in safety_requests.items():
    (root/f"request_{k}.json").write_text(json.dumps(req))
print("Final plan:",{k:v for k,v in plan.items() if k not in ("camera_panels","safety_requests")})
with (root/"notes.md").open("a") as f:
    f.write("\n## Passes26-27: attention ablation not established; stop hypothesis search\n"
            "- Ablating64 query-gain coords returns suspect coupling norms to28-39 vs original900-1067, cap4 106-136.\n"
            "- Attention entropy rises .152 -> .398; H7 agent fraction .999 -> .162, other-head maps nearly unchanged.\n"
            "- Action camera discrepancies essentially unchanged. Discovery64/128 ->66/128 (+1.563pp SE2.93). Reverted.\n"
            "- Final evaluation returns to the ORIGINAL frozen readout_camera_rank2_half tensor, no further tuning.\n"
            "- 7 more paired camera panels (896 episodes) plus prior474 =1370, 602 selection-disjoint variants, 2-3 seeds each. Passes28,29,30,31,38,39,40.\n"
            "- Passes32-37: all6 noncamera categories,128 paired episodes each.\n"
            "- Final retention criterion fixed in final_evaluation_plan.json: camera gain >1.96*variant-cluster SE, no noncamera point-estimate loss >2pp; otherwise ORIGINAL.\n"
            "- No active edit at this moment; restore frozen candidate exactly from archive before pass28.\n")
```
