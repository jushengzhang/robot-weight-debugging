The late expert does not erase the instruction difference: its absolute size stays roughly constant while the residual norm doubles at layer 17. I also noticed unusually large adaptive-norm shift biases at two previously edited layers. I will verify the weight-restoration bookkeeping before interpreting further experiments, since a stale diagnostic edit would invalidate the comparisons.
```python
qshift=torch.load(W/"subspaces"/"query_content_shift.pt",map_location="cpu",weights_only=False)
print("Shift artifact info",str(qshift.get("info"))[:1200])
print("Saved candidate files:")
for p in (W/"candidate_query_content_d1").iterdir():
    c=torch.load(p,map_location="cpu",weights_only=False)
    print(p.name,type(c),tuple(c.shape) if torch.is_tensor(c) else str(c)[:100])
    if torch.is_tensor(c):
        n=p.name.removesuffix(".pt")
        if n not in NAMES:
            matches=[x for x in NAMES if x.replace(".","_")==n]
            print("name lookup",matches[:1]); continue
        a=load_original(n)
        print("original-vs-candidate norm",float((a-c.float()).norm()),
              "max",float((a-c.float()).abs().max()))
print("Shift keys",list(qshift["shifts"]))
for l,s in qshift["shifts"].items():
    print("shift",l,tuple(s.shape),round(float(s.norm()),4))
for dirname in ["candidate_l14_register_head_half","candidate_controlled_visualV_g2",
                "candidate_grounding_heads_g15","candidate_bos_key_half"]:
    p=next(iter((W/dirname).iterdir()))
    n=p.name.removesuffix(".pt")
    if n in NAMES:
        c=torch.load(p,map_location="cpu",weights_only=False).float()
        a=load_original(n)
        print(dirname,"relative diff original",round(float((a-c).norm()/a.norm()),6))
print("Edited list",edited())
```
