I will compare attention-output divergence with residual divergence and examine the stored action traces for failed episodes. This should separate instruction-related grounding errors from failures that only emerge later during grasping or multi-step execution.
```python
pr6=json.loads((W/"probe_6/results.json").read_text())
zz=np.load(pr6[0]["activations"])
print("Attention-output keys:",zz.files[:6])
A=np.stack([np.stack([np.load(r["activations"])[f"mean::action.{l}.attn_out::action"]
                     for l in range(30)]) for r in pr6])
AD=A[1::2]-A[::2]
print("Layer  attn_diff_RMS  attn_good_RMS  residual_increment_RMS")
for l in range(30):
    inc=D[:,l]-(D[:,l-1] if l else 0)
    print(f"{l:2d}  {np.sqrt(np.mean(AD[:,l]**2)):12.5f}  "
          f"{np.sqrt(np.mean(A[::2,l]**2)):12.5f}  {np.sqrt(np.mean(inc**2)):12.5f}")
print("\nStored actions: summary per paired episode")
for i,p in enumerate(pairs):
    ss=[]
    for role in ["good","bad"]:
        ar=np.load(p[role]["actions_npy"])
        if ar.ndim!=2: ar=ar.reshape(-1,7)
        ss.append({"result":role,"steps":p[role]["steps"],
                   "xyz_rms":round(float(np.sqrt(np.mean(ar[:,:3]**2))),3),
                   "last_xyz_rms":round(float(np.sqrt(np.mean(ar[-50:,:3]**2))),3),
                   "gripper_closed_frac":round(float((ar[:,-1]>0).mean()),3)})
    print(i, ss)
print("\nUnmodified input contents (types/shapes, not pixels):")
inp=torch.load(pr4[0]["input"],map_location="cpu",weights_only=False)
print({k:(type(v).__name__,list(v.shape) if hasattr(v,"shape") else str(v)[:180])
       for k,v in inp.items()})
np.savez(W/"paired_attn_diagnosis.npz",outputs=A,difference=AD)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 5-6 diagnosis\n"
            "- Pass 5 attention maps expose action-to-video only; two identical-observation instruction pairs show large visual-attention redistribution at L10-15 (individual head L1 changes up to 0.96). Not causal yet.\n"
            "- Pass 4 residual contrasts grow sharply at L11 (increment RMS 0.0723 vs L10 0.0227); relative residual difference reaches ~10% late. L13 H14 static OV gain is broad-spectrum, not a rank-one sink. Its top six action OV modes explain only 0.53% of L13 residual contrast increment, so a rank-one edit there is not supported.\n"
            "- Pass 6 records original attention outputs for all ten natural instruction pairs, all layers. No active edits. 6/40 passes used.\n")
```
