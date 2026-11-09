The late visual-offset edit produced no net gain, so I will remove it. I will next test the earlier observation that several expert heads are almost entirely wrist-directed. The proposed intervention will remove part of a wrist-common key component, preserving the measured base-camera and language means, rather than weakening all wrist information.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass18: F rejected\n71->71/128, 0pp SE2.2; env+1, object-1; task/position/semantic unchanged; old-goal11 unchanged, loops30->32. Removed. Large late visual mean alone is not sufficient causal explanation.\n")
wrist_key_bases={}
print("Wrist-common pre-LN contrast after base/language protection")
for l in [4,6,8,10,12]:
    ln=load_original(f"{VLM}.layers.{l}.input_layernorm.weight")
    def xn(group):
        x=mat(a14,f"mean::vlm.{l-1}.resid::{group}")
        return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()*(1+ln)
    base,wr,lang=xn("base"),xn("wrist"),xn("lang")
    q=ortho_columns(torch.cat([base.T,lang.T,torch.eye(2048)[:,[50,604,674,738]]],1))
    mu=wr.mean(0); u=mu-q@(q.T@mu); retention=float(u.norm()/mu.norm()); u=u/u.norm()
    wrist_key_bases[l]=u
    v=load_original(f"{VLM}.layers.{l}.self_attn.k_proj.weight")
    print(l,"retained common",round(retention,3),
          "key change wrist ratio",round(float((wr@u[:,None]@(u[None,:]@v.T)).norm()/(wr@v.T).norm()),3),
          "base/lang leakage",round(float((torch.cat([base,lang])@u).norm()/torch.cat([base,lang]).norm()),7))
torch.save(wrist_key_bases,W/"wrist_key_bases.pt")
for k in [2,10]:
    results=json.loads((W/f"probe_{k}"/"results.json").read_text())
    print("probe",k, str(results[0])[:1800])
# Candidate G: attenuate wrist-common key response at L8,12 by half.
for l in [8,12]:
    n=f"{VLM}.layers.{l}.self_attn.k_proj.weight"
    v=load_original(n); u=wrist_key_bases[l]
    save_edit(n,v-0.5*torch.outer(v@u,u))
print("G edit relative changes",[(n.split(".layers.")[1].split(".")[0],round(float((load(n)-load_original(n)).norm()/load_original(n).norm()),4)) for n in edited()])
```
