I will quantify how much the noun swap changes translation, rotation, and gripper output, then locate where its effect grows or shrinks inside the expert. I will also restore the original embeddings immediately after this diagnostic.
```python
for n in list(edited()): drop_edit(n)
pr[20]=json.loads((W/"probe_20/results.json").read_text())
p20={(x["idx"],x["arm"]):x for x in pr[20]}
swap_act={}
for i in range(8):
    xo,xe=p20[i,"orig"],p20[i,"edit"]
    ao,ae=np.load(xo["action"]),np.load(xe["action"])
    A=dict(np.load(xo["activations"])); B=dict(np.load(xe["activations"]))
    swap_act[i]=(A,B)
    d=ae-ao
    print("input",i,Path(xo["input"]).name,
          "action RMS xyz/rot/grip",[round(float(np.sqrt(np.mean(d[:,sl]**2))),5)
                                   for sl in [slice(0,3),slice(3,6),slice(6,7)]])
    if i in [0,5]:
        print(" first XYZ orig/swap:",[round(float(x),4) for x in ao[0,:3]],
              [round(float(x),4) for x in ae[0,:3]])
        print(" VLM relative difference lang/base:")
        for l in [0,1,2,3,6,9,12,15,17]:
            vals=[]
            for m in ["lang","base"]:
                key=f"mean::vlm.{l}.resid::{m}"
                vals.append(float(np.linalg.norm(B[key]-A[key])/(np.linalg.norm(A[key])+1e-8)))
            print(l,[round(v,4) for v in vals])
        print(" Expert relative difference per layer:")
        out=[]
        for l in range(18):
            key=f"calls::expert.{l}.resid::action"
            aa,bb=A[key],B[key]
            # First, middle and last denoising call.
            out.append((l,[round(float(np.linalg.norm(bb[t]-aa[t])/
                                      (np.linalg.norm(aa[t])+1e-8)),4) for t in [0,4,9]]))
        print(out)
print("Active edits:",edited())
```
