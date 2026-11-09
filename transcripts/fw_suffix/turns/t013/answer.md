I will quantify the controlled raw–clean differences and check where they first appear. In particular, whether video layer 0 changes will help determine what the tool’s generic `attn_out` hook captures. I will then make a reversible, half-strength intervention on the video expert’s text cross-attention outputs.
```python
p4 = json.loads((work/"probe_4/results.json").read_text())
print("Probe 4 first result:", str(p4[0])[:1500])
with np.load(p4[0]["activations"]) as a:
    print("Activation keys/shapes:", [(k,a[k].shape) for k in a.files[:6]])

def probe_arrays(pass_no, field="action_norm"):
    res = json.loads((work/f"probe_{pass_no}/results.json").read_text())
    return np.stack([np.load(x[field]) for x in res])
base_actions = probe_arrays(4)
base_raw = base_actions[0::2]
base_clean = base_actions[1::2]
print("Reference action_norm paired RMS:", round(float(np.sqrt(np.mean((base_raw-base_clean)**2))),5))
print("Reference action_norm RMS:", round(float(np.sqrt(np.mean(base_clean**2))),5))

probe4_acts = [dict(np.load(r["activations"])) for r in p4]
layer_diffs = []
for expert in ["video","action"]:
    for l in range(30):
        keys = [k for k in probe4_acts[0] if f"::{expert}.{l}.attn_out::" in k]
        if not keys:
            continue
        a = np.stack([np.concatenate([r[k].reshape(-1) for k in keys]) for r in probe4_acts])
        dr = a[0::2]-a[1::2]
        layer_diffs.append((expert,l,float(np.linalg.norm(dr)/(np.linalg.norm(a[1::2])+1e-12)),
                            float(np.sqrt(np.mean(dr**2)))))
print("Exact-input activation differences (relative Frobenius, difference RMS):")
for x in layer_diffs:
    if x[1] in [0,1,2,3,5,7,10,12,15,20,23,25,29]:
        print(x[0],x[1],round(x[2],4),round(x[3],4))
print("Action->video attention raw-clean change:")
for l in ["0","10","12","23"]:
    vals=[]
    for i in [0,2]:
        ar=np.load(probe3[i]["attention"][l]).astype(float)
        ac=np.load(probe3[i+1]["attention"][l]).astype(float)
        vals.append(np.abs(ar-ac).sum(-1).mean())
    print(l, "mean L1 mass shift", round(float(np.mean(vals)),4))

def clear_edits():
    for n in list(edited()):
        drop_edit(n)

def scale_cross(expert, layers, scale, bias=True):
    for l in layers:
        for tail in (["o.weight","o.bias"] if bias else ["o.weight"]):
            n=f"mot.mixtures.{expert}.blocks.{l}.cross_attn.{tail}"
            save_edit(n, load_original(n)*scale)

clear_edits()
scale_cross("video", range(30), 0.5, bias=True)
print("Current edit:", len(edited()), "tensors, video cross-attention output scale=0.5")
(work/"current_edit.json").write_text(json.dumps(
    {"name":"video_cross_half","expert":"video","layers":list(range(30)),"scale":0.5,"bias":True}))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 4: controlled original reference\n"
            "48 inputs = 24 identical-observation raw/strip pairs, noise_seed=42, all-layer attn_out. "
            f"Action normalized raw-clean RMS={float(np.sqrt(np.mean((base_raw-base_clean)**2))):.6f}.\n"
            "\n### Candidate V0.5\nScale video cross_attn.o weights and biases by 0.5 in all 30 layers. "
            "Action expert, shared attention projections, MLPs, and encoders unchanged. "
            "Temporary causal diagnostic, not yet a validated repair.\n")
```
