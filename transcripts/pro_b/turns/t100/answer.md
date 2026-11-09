I will measure the probe’s attention and action changes before spending a rollout pass. A routing edit that barely changes the predicted action, or mostly strengthens the existing attention sink, would not support the proposed repair.
```python
p29=W/"probe_29"
for i in range(4):
    ao=np.load(p29/f"probe_orig_{i}_action.npy")
    ae=np.load(p29/f"probe_edit_{i}_action.npy")
    print("input",i,"XYZ action RMS",round(float(np.sqrt(np.mean((ao[:,:3]-ae[:,:3])**2))),5),
          "grip RMS",round(float(np.sqrt(np.mean((ao[:,6]-ae[:,6])**2))),5))
    for l in [14,15,17]:
        a=np.load(p29/f"probe_orig_{i}_attn_L{l}.npy").mean(1)
        b=np.load(p29/f"probe_edit_{i}_attn_L{l}.npy").mean(1)
        print(" L",l,"TV",round(float(np.abs(a-b).sum(1).mean()/2),3),
              "BOS",round(float(a[:,768].mean()),3),"->",round(float(b[:,768].mean()),3))
        if l==14:
            for h in [0,1,3]:
                peaks0=np.argsort(a[h,:512])[-2:][::-1]
                peaks1=np.argsort(b[h,:512])[-2:][::-1]
                print("  H",h,"TV",round(float(np.abs(a[h]-b[h]).sum()/2),3),
                      "peak",peaks0.tolist(),"->",peaks1.tolist())
# Inspect motion/noise-padding structure, without evaluating the network.
ain=load_original("action_in_proj.weight")
aout=load_original("action_out_proj.weight")
print("Action input column norms",ain.norm(dim=0).round(decimals=3).tolist())
print("Action output row norms",aout.norm(dim=1).round(decimals=3).tolist())
print("Action biases first10",load_original("action_out_proj.bias")[:10].round(decimals=4).tolist())
```
