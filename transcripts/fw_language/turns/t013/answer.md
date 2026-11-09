Channel 1938 is written most strongly by text attention, but that still does not establish that it causes language failures. I will finish the static analysis by checking text-embedding anisotropy and prepare task matching for canonical-instruction controls, then allow the backend some startup time.
```python
t0=time.time()
base_tasks=sorted({(v["suite"],v["name"].split("_language_")[0]) for v in lang_variants})
def task_of(v):
    matches=[b for s,b in base_tasks if s==v["suite"] and
             (v["name"]==b or v["name"].startswith(b+"_"))]
    return max(matches,key=len) if matches else None
by_task_cat=collections.defaultdict(list)
for v in variants:
    by_task_cat[(v["suite"],task_of(v),v["category"])].append(v)
print("Canonical tasks:",len(base_tasks))
print("Tasks with language and difficulty-1 light controls:",
      sum(any(v["difficulty"]==1 for v in by_task_cat[(s,b,"Light Conditions")])
          for s,b in base_tasks))

text_spectra={}
for expert in ("action","video"):
    for idx in (0,2):
        x=load_original(f"mot.mixtures.{expert}.text_embedding.{idx}.weight")
        s=torch.linalg.svdvals(x)
        text_spectra[f"{expert}_{idx}"]=s.numpy()
        print(expert,"text layer",idx,"top singular values",
              [round(float(v),3) for v in s[:6]],"median",round(float(s.median()),3),
              "top1 energy",round(float(s[0]**2/s.square().sum()),4))
Vsign=V.clone()
Vsign*=torch.sign(Vsign@Vsign[0]).unsqueeze(1)
_,sc,vct=torch.linalg.svd(Vsign,full_matrices=False)
common_context=vct[0]
E2=load_original("mot.mixtures.video.text_embedding.2.weight")
gain=(E2.T@common_context).norm()
rms=(E2.square().sum()/E2.shape[0]).sqrt()
print("Common context mode PC energy:",round(float(sc[0]**2/sc.square().sum()),3),
      "text layer2 gain / random RMS:",round(float(gain/rms),3))
np.savez(W/"text_weight_spectra.npz",**text_spectra)
torch.save(common_context,W/"video_common_context_mode.pt")
(W/"task_index.json").write_text(json.dumps([{"suite":s,"task":b} for s,b in base_tasks]))
print("Analysis seconds:",round(time.time()-t0,1))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass plan\n"
            "Aim to spend <=16 passes on diagnosis and causal discovery, then lock "
            "the edit for remaining paired language and six-category validation. "
            "Exclude all discovery variants from novel-variant language estimates. "
            "If evidence is weak, retain the original rather than an unjustified edit.\n")
time.sleep(max(0,60-(time.time()-t0)))
print("Static screening complete; awaiting behavioral evidence.")
```
