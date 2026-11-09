I will summarize attention by head and token group, including entropy and dominant image patches, and compare the saved poses and action chunks. This distinguishes attention to an unhelpful visual region from an unconditional gripper oscillation; neither explanation is established yet.
```python
p2=json.loads((W/"probe_2/results.json").read_text())
print("probe result format:",str(p2)[:5500])
print("saved state and trace summaries:")
for sid,g in [(647,"close"),(647,"open"),(1300,"close"),(1300,"open"),(322,"open")]:
    r=R[sid,g]; acts=np.load(r["actions_npy"])
    pp=sorted(pathlib.Path(r["frames_dir"]).glob("*inputs.pt"))
    states=[torch.load(p,map_location="cpu",weights_only=False)["element"]["observation/state"] for p in pp]
    print(sid,g,"success",r["success"],"state xyz at 0,50,last",
          np.array(states)[[0,min(1,len(states)-1),-1],:3].round(3).tolist(),
          "action mean(abs xyz/rot/grip)",np.abs(acts).mean(0).round(3).tolist())
    flips=np.where(np.sign(acts[1:,6])!=np.sign(acts[:-1,6]))[0]+1
    print(" flip times",flips[:22].tolist(),"last100 xyz mean",acts[-100:,:3].mean(0).round(3).tolist())
```
