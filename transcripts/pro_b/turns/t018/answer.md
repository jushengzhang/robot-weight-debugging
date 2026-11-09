Original-task sampling is not available through this practice interface; I’ll document that limitation rather than claim an unmeasured guarantee. I’m reverting the key edit and shifting to environment sensitivity, where the observations show a large background change despite unchanged objects and instructions.

```python
clear_edits()
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 5: rejected key-only edit\n"
            "At the failed tomato-sauce input, layer6 content attention 3.04% ->1.82%, "
            "BOS38.48% ->46.03%; initial action RMS change only0.0058. "
            "No predicted routing improvement; reverted without rollout.\n"
            "Original-task sample request refused (no matching practice episodes, did not count). "
            "Cannot directly verify original-task performance with the available listed episodes; "
            "must use object/semantic regression checks and disclose this limitation.\n")
# Quantify already-observed environment contrasts, using initial views only.
env_pairs=[(34,17),(35,11),(36,3),(37,5)]
for branch,layer,group in (("vision",26,"base"),("vision",26,"wrist"),
                            ("vlm",6,"base"),("vlm",6,"lang")):
    x=stack3(layer,group,branch)
    d=torch.stack([x[a]-x[b] for a,b in env_pairs])
    _,s,_=torch.linalg.svd(d,full_matrices=False)
    print(branch,layer,group,"delta/clean norm",
          [round(float((x[a]-x[b]).norm()/x[b].norm()),3) for a,b in env_pairs],
          "singular energy",[round(float(v),3) for v in s.square()/s.square().sum()])
# Analyze positional weights, without executing the vision tower.
pos=load_original(f"{VIS}.embeddings.position_embedding.weight")
_,ps,_=torch.linalg.svd(pos-pos.mean(0),full_matrices=False)
print("position embedding norms min/median/max",
      [round(float(v),2) for v in (pos.norm(dim=1).min(),pos.norm(dim=1).median(),pos.norm(dim=1).max())])
print("position top singular energy", [round(float(v),3) for v in (ps[:8].square()/ps.square().sum())])
print("current edits:",edited())
```
