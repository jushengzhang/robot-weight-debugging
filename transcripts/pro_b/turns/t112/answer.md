I will summarize the positional intervention’s attention changes, including mass on the five affected patches and the predicted motion. The important question is whether it changes the position-dependent route without producing a large nonspecific control disturbance.
```python
p34=W/"probe_34"
affected=np.array(rows+ [x+256 for x in rows])
for i in range(4):
    ao=np.load(p34/f"probe_orig_{i}_action.npy")
    ae=np.load(p34/f"probe_edit_{i}_action.npy")
    print("input",i,"action XYZ RMS",round(float(np.sqrt(np.mean((ao[:,:3]-ae[:,:3])**2))),5),
          "rotation RMS",round(float(np.sqrt(np.mean((ao[:,3:6]-ae[:,3:6])**2))),5),
          "grip RMS",round(float(np.sqrt(np.mean((ao[:,6]-ae[:,6])**2))),5))
    for l,h in [(3,3),(9,2),(14,0),(14,1),(14,3)]:
        a=np.load(p34/f"probe_orig_{i}_attn_L{l}.npy")[h].mean(0)
        b=np.load(p34/f"probe_edit_{i}_attn_L{l}.npy")[h].mean(0)
        print(" ",l,h,"TV",round(float(np.abs(a-b).sum()/2),3),
              "affected mass",round(float(a[affected].sum()),3),"->",round(float(b[affected].sum()),3),
              "peak",int(a[:512].argmax()),"->",int(b[:512].argmax()))
print("Current edits",edited())
```
