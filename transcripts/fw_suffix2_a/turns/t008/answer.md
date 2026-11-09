The policy prompt includes a fixed video-description prefix, and the old candidate did edit both experts. I will collect a controlled original-model reference on the saved first observations, using one fixed diffusion-noise seed, so candidate action changes can be compared directly with the clean-instruction target.
```python
import shutil
replay = W/"replay_diag"
replay.mkdir(exist_ok=True)
def source_input(rid,p):
    return W/f"roll_{rid}_frames"/f'{p["suite"]}_v{p["id"]}_s{p["seed"]}_original'/"t0000_inputs.pt"
replay_meta = []
pair_equal = []
for label,rid,items in (("r",1,splits["diagnostic"]),("c",2,splits["diagnostic"]),("l",1,splits["language_diag"])):
    for j,p in enumerate(items):
        src=source_input(rid,p)
        dst=replay/f"{label}{j:03d}.pt"
        shutil.copyfile(src,dst)
        replay_meta.append(dict(label=label,j=j,path=str(dst),**p))
for p in splits["diagnostic"]:
    a=torch.load(source_input(1,p),map_location="cpu",weights_only=False)
    b=torch.load(source_input(2,p),map_location="cpu",weights_only=False)
    pair_equal.append(torch.equal(a["input_image"],b["input_image"]) and torch.equal(a["proprio"],b["proprio"]))
(W/"replay_meta.json").write_text(json.dumps(replay_meta,indent=2))
print("MATCHED INITIAL OBSERVATIONS:",sum(pair_equal),"/",len(pair_equal))
print("REPLAY DIRECTORY:",str(replay),"files",len(list(replay.glob("*.pt"))))
print("EXAMPLE PROMPTS:")
for label in ("r","c","l"):
    obj=torch.load(replay/f"{label}000.pt",weights_only=False)
    print(label,obj["prompt"])
# Recover only the old detector's explicit direction; do not apply its gate.
a0=torch.from_numpy(np.load(W/"prior_candidate"/"mot.mixtures.action.text_embedding.0.weight.npy"))
u_prior=a0[296].clone()
gain_prior=u_prior.norm().item()
u_prior/=gain_prior
print("PRIOR DETECTOR GAIN:",gain_prior,"bias:",float(np.load(W/"prior_candidate"/"mot.mixtures.action.text_embedding.0.bias.npy")[296]))
torch.save({"u":u_prior,"gain":gain_prior},W/"prior_detector.pt")
```
