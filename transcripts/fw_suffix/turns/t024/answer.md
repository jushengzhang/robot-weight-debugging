I will summarize the direct action-head ablation and record all tested candidates before proceeding. I will also inspect text-projection weight spectra and column-norm concentration to see whether the shared text bottleneck offers a more selective intervention than removing attention heads.
```python
summary8=summarize_probe(8)
A8=probe_arrays(8)
print("Action H6 arm raw->orig-clean:",round(rms(A8[0::2,:,:6]-base_clean[:,:,:6]),5),
      "arm clean drift:",round(rms(A8[1::2,:,:6]-base_clean[:,:,:6]),5),
      "first10 raw->orig-clean:",round(rms(A8[0::2,:10]-base_clean[:,:10]),5))
print("Text projection norm concentration:")
text_weight_stats={}
for e in ["video","action"]:
    for block in [0,2]:
        n=f"mot.mixtures.{e}.text_embedding.{block}.weight"
        W=load_original(n)
        col=W.norm(dim=0)
        row=W.norm(dim=1)
        ix=col.argsort(descending=True)[:8]
        print(e,block,"shape",tuple(W.shape),
              "col max/median",round(float(col.max()/col.median()),3),
              "row max/median",round(float(row.max()/row.median()),3),
              "top columns",ix.tolist())
        if block==0:
            s=torch.linalg.svdvals(W)
            text_weight_stats[e]={"singulars":s}
            print("Singular top12:",[round(float(x),3) for x in s[:12]],
                  "top32 energy",round(float(s[:32].square().sum()/s.square().sum()),4))
clear_edits()
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 8: action_cross_H6 controlled probe\n"+json.dumps(summary8)+"\n"
            "All trial edits cleared after this comparison. No candidate yet validated by rollout.\n")
print("Current edits:",len(edited()))
```
