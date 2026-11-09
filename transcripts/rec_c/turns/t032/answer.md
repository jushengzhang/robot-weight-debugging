Amplifying the base-reading heads also gives no supported gain (-0.8 points, 2.6-point SE), so I will discard it. I will now examine a more localized possibility suggested by the adaptive norms: a few very large residual channels may dominate normalization and drown out corrective visual information. I will compare those channels in successful and looping queries before editing them.

```python
p13=paired_roll(13)
clear_edits()
print("largest residual coordinates in fixed queries (0 success;1-3 failures):")
for l in [0,1,2,7,8,14,15,16,17]:
    print("layer",l)
    for i in range(4):
        v=B8[i][f"mean::expert.{l}.resid::action"]
        ix=np.argsort(np.abs(v))[-4:][::-1]
        share=float(np.square(v[ix]).sum()/(np.square(v).sum()+1e-12))
        print(i,"norm",f"{np.linalg.norm(v):.1f}","top",[(int(j),round(float(v[j]),1)) for j in ix],
              "top4 energy",f"{share:.3f}","ch681",f"{v[681]:.2f}")
print("MLP episode summaries, selected residual-coordinate statistics:")
for l in [5,6,7,8,9,10,14,15,16]:
    k=f"mean::expert.{l}.mlp_out::action"
    ss=np.stack([A4[(r["id"],r["grip"])][k] for r in valid4 if r["success"] and not r.get("success_at_t0")])
    ff=np.stack([A4[(r["id"],r["grip"])][k] for r in valid4 if not r["success"] and r["grip_flips"]>=6])
    ix=np.argsort(np.abs(ff.mean(0)-ss.mean(0)))[-4:][::-1]
    print(l,"norm S/F",f"{np.linalg.norm(ss,axis=1).mean():.1f}/{np.linalg.norm(ff,axis=1).mean():.1f}",
          "diff top",[(int(j),round(float(ss[:,j].mean()),2),round(float(ff[:,j].mean()),2)) for j in ix],
          "681 S/F",f"{ss[:,681].mean():.2f}/{ff[:,681].mean():.2f}")
print("residual VLM dominant coords:")
for l in [0,2,8,14,15,17]:
    k=f"mean::vlm.{l}.resid::wrist"
    a=np.stack([b[k] for b in B8[:4]])
    ix=np.argsort(np.mean(a*a,axis=0))[-3:][::-1]
    print(l,[(int(j),[round(float(v),1) for v in a[:,j]]) for j in ix])
with open(W/"notes.md","a") as f:
    f.write("\n## Pass13: amplify base-reading heads x2\n70/126 edited vs71/126 original (-0.8+/-2.6pp SE),5 wins/6 losses, loops48 vs44. REJECTED. Original baseline varies by1 outcome vs earlier passes despite same declared setup; use within-pass paired results and confirm over independent scenes/seeds.\n")
```
