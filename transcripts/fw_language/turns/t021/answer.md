I will verify the intervention algebra without activating any edit: a half-strength template should halve exactly one effective OV singular mode and preserve the others. This is a safety check on the proposed causal experiment, followed by a longer backend-startup pause.
```python
key="video_L23_H3"
t=templates[key]
Oall=load_original(t["name"])
O=Oall[:,t["head"]*128:(t["head"]+1)*128].clone()
Vh=load_original(t["name"].replace(".o.weight",".v.weight"))[
    t["head"]*128:(t["head"]+1)*128]
u=t["factors"][0]["u"]
r=t["factors"][0]["r"]
D=-0.5*u[:,None]*r[None,:]
Otest=O+D
def product_singular_values(A,B):
    _,Ra=torch.linalg.qr(A,mode="reduced")
    _,Rb=torch.linalg.qr(B.T,mode="reduced")
    return torch.linalg.svdvals(Ra@Rb.T)
sbefore=product_singular_values(O,Vh)
safter=product_singular_values(Otest,Vh)
print("Template tested:",key)
print("Leading singular before/after:",round(float(sbefore[0]),5),round(float(safter[0]),5))
print("Max absolute change of other singular values:",float((sbefore[1:]-safter[1:]).abs().max()))
print("Off-target output-direction leakage:",
      float((D-u[:,None]*(u@D)[None,:]).norm()/D.norm()))
print("Active edits:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n- Algebraic template safety check: half-strength L23 H3 rank-one "
            "projection halves its leading effective OV singular value while preserving "
            "the remaining spectrum to float32 precision. This is not a behavioral test. "
            "No active edits. Simulator still unavailable; waiting for startup.\n")
time.sleep(840)
print("Longer backend startup pause finished.")
```
