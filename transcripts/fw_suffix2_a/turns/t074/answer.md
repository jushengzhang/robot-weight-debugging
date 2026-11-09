The development rollout improved success from 47/96 to 52/96: +5.2 points with a paired SE of 3.4 points. That is encouraging but not conclusive. I’ll save this candidate, then test video-only and action-only gating on the same episodes to identify which route is helping.
```python
candidate_dir=W/"candidate_dense_gate_t020_both"
candidate_dir.mkdir(exist_ok=True)
for n in edited():
    np.save(candidate_dir/(n+".npy"),load(n).numpy())
torch.save(torch.load(W/"dense_gate_metadata.pt",weights_only=False),candidate_dir/"metadata.pt")
print("Saved candidate:",candidate_dir.name,len(list(candidate_dir.glob("*.npy"))))
print("Rollout21 first record:",json.loads((W/"roll_21.jsonl").read_text().splitlines()[0]))
print("Exact q/k-normalization bucket sizes (possible lower-loss gate coordinates):")
for ex in ["action","video"]:
    counts=[]
    for layer in [0,10,20,29]:
        q=load_original(f"mot.mixtures.{ex}.blocks.{layer}.cross_attn.norm_q.weight").reshape(24,128)
        k=load_original(f"mot.mixtures.{ex}.blocks.{layer}.cross_attn.norm_k.weight").reshape(24,128)
        ss=[]
        for h in range(24):
            _,co=torch.unique(torch.stack([q[h],k[h]],dim=1),dim=0,return_counts=True)
            ss.append(int(co.max()))
        counts.append((layer,min(ss),round(float(np.mean(ss)),1),max(ss)))
    print(ex,counts)
# Isolate the indirect, video-to-action route; action expert returns exactly to original.
for n in list(edited()):
    if ".action." in n:
        drop_edit(n)
print("Video-only active tensors:",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass21 development rollout, dense six-detector both-expert gate\n96 paired:47->52 successes, +5.21pp, SE3.4pp,8 gains/3 losses. "
            "Background+2,Camera0,Light-2,Layout+1,Initial+3,Noise+1 episodes. "
            "Not conclusive. Immutable copy candidate_dense_gate_t020_both. Video-only causal ablation now active for pass22.\n")
```
