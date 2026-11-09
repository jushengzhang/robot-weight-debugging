The probe exposes modality averages, not full token residuals, so it cannot establish which visual hotspot carries a shortcut. I will begin with a cleaner causal test: increase only layer 9 head 4's output contribution by 50%, because that head reads the final instruction token. All other heads, action scales, and inference settings will remain unchanged.
```python
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def set_head_gains(spec):
    """spec: (layer,head,gain); always reconstruct from original weights."""
    grouped=collections.defaultdict(list)
    for l,h,g in spec:
        grouped[l].append((h,g))
    for l,hgs in grouped.items():
        n=f"{expert_root}.layers.{l}.self_attn.o_proj.weight"
        v=load_original(n).clone()
        for h,g in hgs:
            v[:,256*h:256*(h+1)] *= g
        save_edit(n,v)
def snapshot(label):
    d=W/label
    d.mkdir(exist_ok=True)
    names=list(edited())
    for i,n in enumerate(names):
        torch.save(load(n),d/f"{i:03d}.pt")
    (d/"manifest.json").write_text(json.dumps(names))
def restore(label):
    clear_edits()
    d=W/label
    for i,n in enumerate(json.loads((d/"manifest.json").read_text())):
        save_edit(n,torch.load(d/f"{i:03d}.pt",map_location="cpu",weights_only=True))
clear_edits()
set_head_gains([(9,4,1.5)])
snapshot("candidate_L9H4_150")
print("Edited:",edited())
n=list(edited())[0]
x=load(n); x0=load_original(n)
print("Relative Frobenius change:",float((x-x0).norm()/x0.norm()))
print("Changed entries:",int((x!=x0).sum()),"/",x.numel())
# Compact, correctly labeled residual feature observations.
print("Averaged residual dominant channels:")
for l in [0,9,12,15,16,17]:
    vals={}
    for modality in ["base","wrist","lang"]:
        v=p3a[f"mean::vlm.{l}.resid::{modality}"].astype(np.float64)
        ii=np.argsort(v*v)[-3:][::-1]
        vals[modality]=[(int(j),round(float(v[j]),1),round(float(100*v[j]*v[j]/(v@v)),1)) for j in ii]
    print(l,vals)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 2-3: diagnostic probes\n"
            "Expert L9H4 attends final newline 47-63% across four first observations; "
            "L9H2 attends wrist patch (13,3) 75-81%, total wrist 98.6-99.5%. "
            "Other language-heavy heads often attend BOS, so total language mass is not content attention. "
            "Position1896/1898 move milk 0.353/0.376m, tomato unchanged; trajectories mimic familiar tomato reach. "
            "Probe residuals are modality means, not token-resolved. Final VLM layer output is not itself cached, "
            "so large L17 residual norms do not identify a causal fault.\n"
            "Candidate A: only expert L9H4 o_proj columns 1024:1280 multiplied by 1.5. "
            "Hypothesis: strengthen instruction summary versus scene habitual action; no global scaling.\n")
```
