Neither branch of head 14 produced a net gain, so I will discard that hypothesis. I will now examine the alternative route identified in the static analysis: a large text-written video feature at channel 1938 and the shared-attention heads that read it. The key question is whether its changes track instruction wording rather than ordinary visual task variation.
```python
clear_edits()
print("Active edits:",edited())
print("\nStatic reader-screen summary:")
cr=json.loads((W/"common_mode_read_routes.json").read_text())
print(compact_obj(cr,3000))
for fn in ["video_common_mode.pt","rank_one_intervention_templates.pt"]:
    ob=torch.load(W/fn,map_location="cpu",weights_only=False)
    print(fn,"keys",list(ob)[:12])
    if fn.startswith("rank"):
        kk=next(iter(ob)); print("Template example",kk,
              {k:(tuple(v.shape) if torch.is_tensor(v) else v) for k,v in ob[kk].items()})

def means_of(rec,l):
    z=np.load(rec["activations"])
    ka=f"mean::video.{l}.resid::agent"
    kw=f"mean::video.{l}.resid::wrist"
    return .5*(z[ka]+z[kw]) if ka in z else None
vl=[0,8,12,13,15,17,20,22,23,24,27,29]
print("\nWithin-task instruction-pair video contrasts:")
for l in vl:
    g=np.stack([means_of(p["good"],l) for p in pairs])
    b=np.stack([means_of(p["bad"],l) for p in pairs])
    d=b-g
    frac=(d[:,1938]**2).sum()/(d*d).sum()
    print(l,"channel1938 good mean",round(float(g[:,1938].mean()),2),
          "bad mean",round(float(b[:,1938].mean()),2),
          "contrast energy fraction",round(float(frac),4),
          "range",np.round(np.min(g[:,1938]),1),np.round(np.max(g[:,1938]),1))
print("\nSame-task canonical-light vs language means, L23:")
controls={(r["suite"],task_of(r)):r for r in r2}
for p in pairs:
    r=p["bad"];c=controls.get((r["suite"],task_of(r)))
    if c:
        print(r["suite"],r["id"],"canonical",round(float(means_of(c,23)[1938]),2),
              "good",round(float(means_of(p["good"],23)[1938]),2),
              "bad",round(float(means_of(r,23)[1938]),2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 11\n"
            "- Action-only L13 H14 half-gain: original 89/128, edited 89/128; 0.0 pp +/-1.6 SE; 2 rescues, 2 regressions. Reverted. Combined and branch-isolated tests provide no support for this candidate.\n"
            "- Investigating previously identified video-channel-1938 text-write/shared-attention-read route next; no accepted edit yet. 11/40 passes used.\n")
```
