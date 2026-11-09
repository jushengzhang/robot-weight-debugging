The rank-two edit covered only about 26% of the protected camera-contrast energy. I’ll quantify how well those residual contrasts track the actual action errors and test subspace stability across disjoint task groups. This will tell us whether a broader readout intervention is justified before testing it in the simulator.
```python
for n in NAMES:
    if "mixtures.action." in n and ".blocks." not in n:
        print(n, SHAPES[n])
for pname in ("probe_10", "probe_12"):
    obj = json.loads((root / pname / "results.json").read_text())
    print(pname, "results sample:", str(obj)[:1200])

def probe_stack(k, n, layer, what="resid"):
    prefix = "orig"
    return np.stack([np.load(root/f"probe_{k}"/f"probe_{prefix}_{i}_acts.npz")[f"mean::action.{layer}.{what}::action"] for i in range(n)])
R1 = probe_stack(10, 57, 29)
R2 = probe_stack(12, 111, 29)
R = torch.from_numpy(np.concatenate([R1.reshape(19,3,1024), R2.reshape(37,3,1024)])).double()
D = R[:,1] - R[:,0]
Dperp = D - (D @ P) @ P.T
print("Projection consistency, total/perp camera RMS:", float(D.square().mean().sqrt()), float(Dperp.square().mean().sqrt()))
print("SVD stored error:", float(torch.linalg.norm(Dperp.T@Dperp - (Vcam*s.square())@Vcam.T)/torch.linalg.norm(Dperp.T@Dperp)))
task_split = bz["task_split"]
print("Task split values:", np.unique(task_split, return_counts=True))
split_values = np.unique(task_split)
if len(split_values) == 2:
    A = Dperp[task_split == split_values[0]]
    B = Dperp[task_split == split_values[1]]
else:
    A, B = Dperp[::2], Dperp[1::2]
_,sa,va = torch.linalg.svd(A, full_matrices=False)
_,sb,vb = torch.linalg.svd(B, full_matrices=False)
print("Task split stability: rank, cross-energy A->B/B->A, median principal cosine")
for rank in (1,2,4,8,12,16,24):
    rank = min(rank,len(va),len(vb))
    u, v = va[:rank].T, vb[:rank].T
    cap_a = float((B@u).square().sum()/B.square().sum())
    cap_b = float((A@v).square().sum()/A.square().sum())
    co = torch.linalg.svdvals(u.T@v).numpy()
    print(rank, round(cap_a,3), round(cap_b,3), round(float(np.median(co)),3), "min", round(float(co.min()),3))

def original_actions(k, count):
    return np.stack([np.load(root/f"probe_{k}"/f"probe_orig_{i}_action_norm.npy") for i in range(count)])
A1 = original_actions(10, 57)
A2 = original_actions(12, 111)
print("Actions shapes:", A1.shape, A2.shape)
Ac = np.concatenate([A1.reshape(19,3,*A1.shape[1:]), A2.reshape(37,3,*A2.shape[1:])])
delta_action = torch.from_numpy((Ac[:,1,:10]-Ac[:,0,:10]).mean(1)).double()
delta_linear = D @ W0.T
print("Axis action vs residual-readout correlations:",
      [round(float(np.corrcoef(delta_action[:,j], delta_linear[:,j])[0,1]),3) for j in range(7)])
print("Action camera contrast variance accounted in readout-space by rank:")
for rank in (2,8,16,32,56):
    er = (Dperp @ Vcam[:,:rank]) @ (W0 @ Vcam[:,:rank]).T
    print(rank, "pred_linear_fraction", round(float(er.square().sum()/delta_linear.square().sum()),3),
          "corr_to_action", round(float(torch.nn.functional.cosine_similarity(er.flatten(),delta_action.flatten(),dim=0)),3))
```
