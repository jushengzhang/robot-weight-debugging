The early-expert contrasts have a remaining confound: the later comparison used step 50 for successes and step 250 for loops, so their noise samples differ systematically. I will correct that by probing the looping runs at step 50 too. I will also inspect the unusually low-dimensional late-VLM contrasts for attention-sink channels.
```python
d=W/"d20";d.mkdir(exist_ok=True)
mf20=[]
for i in range(0,len(mf),4):
    r=mf[i+3]
    src=Path(r["source"]).parent/"t0050_inputs.pt"
    assert src.exists()
    dest=d/f"{len(mf20):02d}.pt"
    if dest.exists() or dest.is_symlink():dest.unlink()
    dest.symlink_to(src)
    mf20.append({**r,"step":50,"input":str(dest),"source":str(src),
                 "success_index19":i+2,"initial_loop_index19":i+1})
(W/"mlp_time_matched_manifest.json").write_text(json.dumps(mf20,indent=2))
for k in ["mean::vlm.16.mlp_out::base","mean::vlm.16.mlp_out::wrist",
          "mean::vlm.15.mlp_out::wrist","mean::expert.16.mlp_out::action"]:
    x=X[k]; ds=contrast[k]
    print(k)
    for phase in ["initial","later"]:
        dd=ds[phase];inds=np.argsort(dd*dd)[-5:][::-1]
        print(" ",phase,"topdims",inds.tolist(),"energy",
              np.round((dd[inds]**2)/(dd@dd),3).tolist(),
              "contrast/mean cos",round(cos(dd,x.mean(0)),3))
# Metadata only: confirm whether action noise is identical when time and grip match.
def tensor_fields(v,p=""):
    out={}
    if isinstance(v,dict):
        for k,z in v.items():out.update(tensor_fields(z,p+"/"+str(k)))
    elif torch.is_tensor(v) or isinstance(v,np.ndarray):
        out[p]=tuple(v.shape)
    return out
print("input fields",tensor_fields(torch.load(mf20[0]["source"],weights_only=False)))
req={"type":"probe","model":"original","inputs":[r["input"] for r in mf20],
     "return":{"layers":list(range(18)),"vlm_layers":[0,2,6,8,14,15,16,17],"what":"mlp_out"}}
print(json.dumps(req,separators=(",",":")))
```
