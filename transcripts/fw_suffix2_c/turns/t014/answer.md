The probe tool rejected copied inputs because it requires paths registered by a rollout; no pass was charged. I will replace the copies with links to the original files and record the four observation mismatches explicitly. The policy inputs themselves will remain unchanged.
```python
for typ,recs in [("r",raw_recs),("c",clean_recs)]:
    for i,key in enumerate(pairkeys):
        dst=replay_dir/f"{typ}{i:03d}.pt"
        src=Path(recs[key]["frames_dir"])/"t0000_inputs.pt"
        dst.unlink()
        dst.symlink_to(src)
for i,r in enumerate(langs):
    dst=replay_dir/f"l{i:02d}.pt"
    dst.unlink()
    dst.symlink_to(Path(r["frames_dir"])/"t0000_inputs.pt")
print("LINK RESOLVES TO:",(replay_dir/"r000.pt").resolve())
for i,ok in enumerate(pair_ok):
    if not ok:
        x=torch.load(replay_dir/f"r{i:03d}.pt",weights_only=False)
        y=torch.load(replay_dir/f"c{i:03d}.pt",weights_only=False)
        print("EXCLUDE STRICT PAIR",i,pairkeys[i],raw_recs[pairkeys[i]]["category"],
              "image RMS",float((x["input_image"].float()-y["input_image"].float()).square().mean().sqrt()),
              "proprio RMS",float((x["proprio"]-y["proprio"]).square().mean().sqrt()))
with (W/"notes.md").open("a") as f:
    f.write("\nCorrection: 116/120 pairs have exactly identical images and proprio; strict tag-only action metrics must exclude indices "
            +str([i for i,x in enumerate(pair_ok) if not x])+
            ". Copied replay paths refused without charge; replaced by symlinks to registered original files.\n")
```
