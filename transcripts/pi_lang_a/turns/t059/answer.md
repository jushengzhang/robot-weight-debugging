The 16-row baseline gained 7.5 points on the extension panel, but the two targeted failures remain. I will add “spread” → cheese and “cup” → mug at half strength, preserving their original embeddings halfway and leaving every other row unchanged.
```python
def install_aliases(manifest):
    T=E0.to(torch.bfloat16)
    for x in manifest:
        assert x["source_id"] not in protected
        alpha=float(x["alpha"])
        T[x["source_id"]]=((1-alpha)*E0[x["source_id"]]+alpha*E0[x["target_id"]]).to(torch.bfloat16)
    save_edit(EMB,T)
    del T

candidate18half=candidate16+[dict(x,alpha=.5) for x in extra]
install_aliases(candidate18half)
(W/"candidate18_half.json").write_text(json.dumps(candidate18half,indent=2))
s26,pairs26=paired_summary(records(26))
print("Baseline16:",s26)
for key in sorted(newkeys):
    subset=[(a,b,d) for a,b,d in pairs26 if (a["suite"],a["id"])==key]
    print(key,"n",len(subset),"original",sum(a["success"] for a,b,d in subset),
          "base16",sum(b["success"] for a,b,d in subset))
with (W/"notes.md").open("a") as f:
    f.write("\nPass26 candidate16 baseline extension-panel80: original49/80 edited55/80,+7.5pp(SE2.96),6rescues0regressions. Now installing candidate18half:16 unchanged full aliases plus spread->cheese and cup->mug at alpha0.5. No training; direct row interpolation only. Aliases exclude every canonical task token.\n")
print("Active rows:",len(candidate18half),"new half-dose rows:",candidate18half[-2:])
```