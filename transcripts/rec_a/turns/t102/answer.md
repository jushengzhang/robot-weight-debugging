I will verify the intended attention change before spending a rollout pass. The edit should reduce the instruction-word fixation without simply suppressing the head or strongly disturbing already visual attention.
```python
pa31=probe_actions(31)
for i in range(4):
    aa=[]
    for arm in ["orig","edit"]:
        m=np.load(W/"probe_31"/f"probe_{arm}_{i}_attn_L16.npy")[3].mean(0)
        aa.append([m[:256].sum(),m[256:512].sum(),m[768:].sum(),1-m.sum()])
    a,b=pa31[i]["orig"],pa31[i]["edit"]
    print(i,"B/W/lang/self",np.round(aa,4).tolist(),
          "XYZ RMS",round(float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))),4),
          "first5 XYZ change",np.round((b[:5,:3]-a[:5,:3]).mean(0),4).tolist(),
          "grip signs",int(np.sum((a[:,6]>0)!=(b[:,6]>0))))
print("New early-head maps:")
for L,hs in [(7,[1,2]),(11,[2,5])]:
    for h in hs:
        masses=[]
        for i in range(4):
            m=np.load(W/"probe_31"/f"probe_orig_{i}_attn_L{L}.npy")[h].mean(0)
            masses.append([m[:256].sum(),m[256:512].sum(),m[768:].sum(),1-m.sum()])
        print(L,h,np.round(masses,3).tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 31: instruction-versus-observation query test\n"
            "- Recovered existing time-conditioning activations from measured adaptive gates; independent gate residual 0.3-0.8%. No weight fitting/training.\n"
            "- E16H3 attends 84.3% to the word 'place' in the alphabet-soup loop, versus 55.8% to a wrist patch in the matched success; not a BOS sink.\n"
            "- Candidate removes one slow-RoPE instruction-vs-image query direction, protected against camera contrast and common key direction.\n")
```
