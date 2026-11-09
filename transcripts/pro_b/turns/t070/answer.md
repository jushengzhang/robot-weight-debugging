Strengthening the layer-14 route did not improve task success, so insufficient signal strength is not supported. I will revert it and test a different possibility: those visual heads may rely too much on fixed relative-position cues. I will attenuate only their fast-rotating RoPE query coordinates, preserving slower content-sensitive coordinates, other heads, and all value projections.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 22: grounded-vision head output gain rejected\n"
            "L14 H0,H1,H3 x1.5 gives60/128->59/128 (-0.8 +/-2.4pp); "
            "task20 unchanged, semantic15->13. Reverted. "
            "The causal language route exists, but increasing its total output does not repair binding.\n")
qname=f"{ex}.layers.14.self_attn.q_proj.weight"
q=load_original(qname).reshape(8,256,1024)
print("L14 query energy by RoPE pair band [0:32,32:64,64:96,96:128]:")
for h in ground_heads:
    E=[]
    for j in range(4):
        sl=slice(j*32,(j+1)*32)
        E.append(float((q[h,sl].square().sum()+q[h,128+j*32:128+(j+1)*32].square().sum())/q[h].square().sum()))
    print(h,[round(v,3) for v in E])
qedit=q.clone()
fast=list(range(64))+list(range(128,192))
for h in ground_heads:
    qedit[h,fast,:] *= 0.5
save_edit(qname,qedit.reshape(2048,1024))
dst=W/"candidate_grounding_rope_half"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
print("Relative weight delta:",round(float((qedit-q).norm()/q.norm()),3))
print("Salad diagnostic episodes:",[(r["id"],r["ptype"],r["instruction"]) for r in r1
                                    if r["task"]=="pick_up_the_salad_dressing_and_place_it_in_the_basket"])
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate fast-RoPE attenuation in grounded visual heads\n"
            "ExpertL14 H0,H1,H3: halve Q rows for RoPE pairs0:64 "
            "(coordinates0:64 and128:192 within each256-D head). "
            "Preserve slow-frequency pairs64:128, all other heads and all K/V/O weights. "
            "Hypothesis: reduce relative-position template dependence in the causally "
            "localized instruction-conditioned visual route; target position robustness.\n")
```
