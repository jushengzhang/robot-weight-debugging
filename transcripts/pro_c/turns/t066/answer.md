Full removal lost the smaller dose’s gain and reduced task success, so I will not retain that dose. I will now stop generating new mechanisms and compare the three modest leads on a fresh paired sample: BOS-key attenuation, anchored illumination attenuation, and the half-dose expert MLP edit. The original development sample has been reused too often to establish a real gain.
```python
for n in list(edited()): drop_edit(n)
def paired_summary(k):
    rows=[json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines()]
    pp=defaultdict(dict)
    for r in rows: pp[(r["id"],r["seed"])][r["model"]]=r
    return {key:(int(v["edited"]["success"])-int(v["original"]["success"])) for key,v in pp.items() if "edited" in v and "original" in v}
print("Development flip overlap among leads")
deltas={k:paired_summary(k) for k in [8,12,21]}
for k,d in deltas.items():
    gains={i for i,x in d.items() if x>0};loss={i for i,x in d.items() if x<0}
    print(k,"gains",sorted(gains),"losses",sorted(loss))
# Restore B0.5 exactly, from originals.
for l in [6,8,10,12]:
    n=f"{VLM}.layers.{l}.self_attn.k_proj.weight"
    v=load_original(n);v[:,[604,738]]*=0.5
    save_edit(n,v)
print("B0.5 restored",edited())
dcdef=torch.load(W/"dc_edit_definition.pt",map_location="cpu",weights_only=False)
print("DC definition keys",list(dcdef))
for k,v in dcdef.items():
    print(k,tuple(v.shape) if hasattr(v,"shape") else str(v)[:250])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass22: reject H full dose\n71->71/128,0pp SE1.9; task8->6, env+1/position+1, other types unchanged. H0.5 remains a weak lead only. No demonstrated repair of original-task override. Stop new mechanism search; independently screen B0.5, D0.5, H0.5, then freeze any supported choice before final validation. Current edit restored B0.5 (K columns604/738 at6/8/10/12).\n")
```
