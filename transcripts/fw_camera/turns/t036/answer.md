The nuisance directions align most with layer 11 head 0 and layer 16 head 1, not with the initially suspicious final-layer heads. I will inspect their actual camera routing and compare them with the final-layer candidates. I will also check whether the saved per-call activations allow us to see when the nuisance grows during denoising.```python
with np.load(p2raw[0]["activations"]) as z:
    print("Per-call activation shapes:",[(k,z[k].shape) for k in z.files[60:63]])

agent_idx=np.array([i for i in range(98) if i%14<7])
wrist_idx=np.array([i for i in range(98) if i%14>=7])
maps3={}
head_map_stats={}
for l in (11,16,22,29):
    m=np.stack([np.load(r["attention"][str(l)]).astype(np.float64) for r in p3])
    maps3[l]=m
    # Mean over 32 action queries; raw mass still includes normalization against action tokens.
    avg=m.mean(2)
    ma=avg[:,:,agent_idx].sum(2)
    mw=avg[:,:,wrist_idx].sum(2)
    tot=ma+mw
    cond=avg/(tot[:,:,None]+1e-12)
    ent=-(cond*np.log(cond+1e-12)).sum(2)/math.log(98)
    tv=np.abs(cond[[1,3]]-cond[[0,2]]).sum(2).mean(0)/2
    wa=ma/(tot+1e-12)
    head_map_stats[l]={"video_mass":tot,"agent_fraction":wa,"entropy":ent,"camera_pair_tv":tv}
    print("\nL%02d: top pair-sensitive heads (head, TV, video mass, agent frac low/high, entropy)"%l)
    for h in np.argsort(tv)[-6:][::-1]:
        print(int(h),round(float(tv[h]),3),np.round(tot[:,h],3).tolist(),
              np.round(wa[:,h],3).tolist(),np.round(ent[:,h],3).tolist())
    if l==29:
        print("Weight-only candidates:")
        for h in (4,7,8,17):
            print(int(h),"TV",round(float(tv[h]),3),"video mass",np.round(tot[:,h],4).tolist(),
                  "agent frac",np.round(wa[:,h],3).tolist(),"entropy",np.round(ent[:,h],3).tolist())
print("\nFocused mid-layer readers:")
for l,h in [(11,0),(11,2),(11,13),(16,1),(16,16),(16,13)]:
    z=head_map_stats[l]
    print(l,h,"video",np.round(z["video_mass"][:,h],3).tolist(),
          "agent",np.round(z["agent_fraction"][:,h],3).tolist(),
          "TV",round(float(z["camera_pair_tv"][h]),3))
np.savez(W/"probe3_head_map_summary.npz",**{
    f"L{l}_{k}":v for l,stats in head_map_stats.items() for k,v in stats.items()
})
```
