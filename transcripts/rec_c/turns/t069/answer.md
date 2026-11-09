Normalization reduces channel 674, but it still contributes a substantial share of the normalized mean representation in early and middle layers. Most of its large offset is already present at layer 0. I will locate that initial source so a test can change one writer rather than suppressing the channel throughout the network.
```python
pfx="paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
pw,pb=load_original(pfx+"weight"),load_original(pfx+"bias")
print("projector row norms 674/50/738",pw[[674,50,738]].norm(dim=1).tolist(),
      "median",float(pw.norm(dim=1).median()),
      "biases",pb[[674,50,738]].tolist())
for L in [0,3,6,9,12,15,17]:
    arr=[]
    for rr in json.loads((W/"roll_1_acts/index.json").read_text()):
        z=np.load(rr["activations"])
        for mod in ["base","wrist","lang"]:
            k=f"mean::vlm.{L}.attn_out::{mod}"
            if k in z:
                m=z[k]
                arr.append([m[674],m[50],np.linalg.norm(m)])
        z.close()
    print("attention out",L,"mean674,50,norm",np.round(np.mean(arr,axis=0),3).tolist())
for L in [0,1,2]:
    a=load_original(f"{vlmprefix}.layers.{L}.self_attn.o_proj.weight")
    print("o_proj L",L,"row norms674/50/738",
          np.round(a[[674,50,738]].norm(dim=1).numpy(),3).tolist(),
          "median",round(float(a.norm(dim=1).median()),3))
del pw,pb,a
```
