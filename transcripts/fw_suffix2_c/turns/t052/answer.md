The combined projection reduces the overall action gap by 18%, but most of that improvement is in the gripper; arm motion changes much less. I’ll test half strength to establish a dose response, and inspect the readout setup before collecting more token-sensitive contrasts.
```python
import shutil
def clear_edits():
    for n in list(edited()): drop_edit(n)
def snapshot(label):
    d=W/label; d.mkdir(exist_ok=True)
    for n in edited():
        np.save(d/(n+".npy"),load(n).numpy())
    return d
snapshot("candidate_video_union_full")
un=torch.load(W/"union_projector.pt",weights_only=False)
tn="mot.mixtures.video.text_embedding.2.weight"
ww=load_original(tn).double()
B,D=un["B"],un["D"]
cand=ww-0.5*(ww@B)@D.T
save_edit(tn,cand.float())
print("HALF DOSE RELATIVE CHANGE",float((cand-ww).norm()/ww.norm()))
print("ACTIVATION KEYS", np.load(W/"probe_10/probe_edit_0_acts.npz").files)
d0=W/"fwd/probe_10_weights"
for l in range(3):
    print("BLOCK",l)
    for part in ["q.bias","norm_q.weight","k.bias","norm_k.weight","q.weight","k.weight","v.weight","o.weight"]:
        aa=torch.from_numpy(np.load(d0/f"mot.mixtures.action.blocks.{l}.cross_attn.{part}.npy"))
        nz=aa.nonzero()
        vals=aa[tuple(nz[:8].T)] if nz.numel() else torch.empty(0)
        print(part,"nonzero",len(nz),"first",nz[:8].tolist(),"vals",vals.tolist(),
              "norm",float(aa.norm()))
with (W/"notes.md").open("a") as f:
    f.write("\nPass16 union projection: gap ratio.8179, arm-first10 .9414, gripper .5008; clean/lang drift.00529/.00567. Saved candidate_video_union_full. Pass17 tests same fixed rank104 directions at half dose.\n")
```
