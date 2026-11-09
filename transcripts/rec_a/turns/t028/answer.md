The saved starts show that the model often copies the visible gripper state, including holding an empty closed gripper before reopening. However, the open/closed pairs also have different saved noise, so their residual differences are not a clean visual intervention. I will quantify that confound and inspect the earliest wrist-focused heads, where the policy may be converting finger appearance into a grasp-state shortcut.
```python
noise_pairs=[]
for i in diag_ids:
    if (i,"open") in initial_obs and (i,"close") in initial_obs:
        noise_pairs.append((np.asarray(initial_obs[(i,"open")]["noise"]),
                            np.asarray(initial_obs[(i,"close")]["noise"])))
Dn=np.stack([(a-b).mean(0) for a,b in noise_pairs])
print("Noise pair mean-direction consistency:",round(float(np.linalg.norm(Dn.mean(0))/np.sqrt((Dn*Dn).sum(1).mean())),3))
print("Distinct initial noise arrays:",len({hashlib.sha256(a.tobytes()).hexdigest() for pair in noise_pairs for a in pair}))
print("Four residual-pair noise identity:",[(
    np.array_equal(initial_obs[(i,"open")]["noise"],initial_obs[(367,"open")]["noise"]),
    np.array_equal(initial_obs[(i,"close")]["noise"],initial_obs[(367,"close")]["noise"])) for i in [367,753,11,1269]])
print("Wrist-focused head top patch coordinates:")
for ii in range(4):
    print("Input",ii)
    for l,h,aa in [(0,6,att3),(3,2,att2),(8,2,att2),(9,2,att5),(15,2,att2)]:
        A=aa[ii][l][h]
        ix=np.argsort(A[:512])[::-1][:4]
        print(f" L{l}H{h}",[("B" if j<256 else "W",int(j%256//16),int(j%16),round(float(A[j]),3)) for j in ix])
# One-head causal test, not a global action/noise rescaling.
cn=head_o_edit(0,6,0.5)
print("Candidate: expert L0 H6 output half-strength")
print("Tensor change:",round(float((load(cn)-load_original(cn)).norm()/load_original(cn).norm()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n## Early gripper-state route hypothesis\n"
            "- Open/closed t0 states differ mainly in fingers, but saved noises also differ: "
            "do not treat their expert residual contrast as purely visual.\n"
            "- L0 H6 is the earliest strongly wrist-focused head (~82-90% wrist mass); "
            "L3 H2 and L15 H2 also focus on finger-region patches.\n"
            "- Test partial L0 H6 O-block attenuation, using identical-input original/edited probes "
            "to identify actual gripper effects before evaluating it as a repair.\n")
```
