Those 64 coordinates account for roughly 89–96% of the query/key gain-product energy in the four suspect heads. I’ll ablate only those coordinates in the action-query normalization, leaving every value projection, output projection, and other head unchanged. This stronger causal test asks whether the remaining coordinate set restores more content-dependent attention.
```python
# Archive/revert the weak readout candidate before a single-site intervention.
for name in list(edited()):
    drop_edit(name)

qW=load_original("mot.mixtures.action.blocks.29.self_attn.q.weight").double()
kW=load_original("mot.mixtures.video.blocks.29.self_attn.k.weight").double()
qzero=qg.clone()
changed_mask=torch.zeros_like(qg,dtype=torch.bool)
coupling_check=[]
for h in outlier_heads:
    mask=prod[h].abs()>4
    changed_mask[h]=mask
    qzero[h,mask]=0
    qa=qW[h*128:(h+1)*128]
    kb=kW[h*128:(h+1)*128]
    gram=(qa@qa.T)*(kb@kb.T)
    p0=prod[h].double()
    pc=p0.clamp(-4,4)
    pz=p0.clone(); pz[mask]=0
    norms=[float(torch.sqrt(torch.clamp(p@(gram@p),min=0))) for p in (p0,pc,pz)]
    coupling_check.append({"head":h,"coupling_fro_original_cap4_zero":norms})
save_edit(qname,qzero.flatten())
ablation_label="l29_qk_outlier64_ablation"
shutil.copytree(root/"edited",root/"candidates"/ablation_label,dirs_exist_ok=True)
(root/"l29_outlier_ablation.json").write_text(json.dumps({
    "heads":outlier_heads,"threshold_original_abs_qk":4,"changed_coordinates":int(changed_mask.sum()),
    "coupling_check":coupling_check},indent=2))
print("Coupling-only analysis:",json.dumps(coupling_check))
print("Active edits:",edited(),"coordinates changed:",int(changed_mask.sum()))
b3=json.loads((root/"probe_3/results.json").read_text())
probe26_inputs=[r["input"] for r in b3]
req26={"type":"probe","model":"edited","inputs":probe26_inputs,
       "return":{"actions":True,"layers":[29],"what":"attn_maps","noise_seed":46001}}
(root/"request_26.json").write_text(json.dumps(req26))
with (root/"notes.md").open("a") as f:
    f.write("\n## New causal test prepared after pass25\n"
            "- Reverted weak revised readout. Active only action L29 self_attn.norm_q.weight.\n"
            "- H4/7/8/17: 64 coordinates with original |q_gain*video_k_gain|>4 account for89-96% of gain-product energy.\n"
            "- Test exact coordinate ablation (set only these q gains to0), stronger than prior cap4; values, output projections and other heads unchanged.\n"
            "- Candidate l29_qk_outlier64_ablation archived. Not yet validated; original fallback.\n")
print("NEXT REQUEST:",json.dumps(req26,separators=(",",":")))
```
