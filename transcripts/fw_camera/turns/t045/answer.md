I will measure the cap’s effect and prepare a direct output test: restore the original query gains, then halve only the output columns belonging to heads 4, 7, 8, and 17 in layer 29. This tests whether their camera-sensitive signals are harmful even after attention is softened.
```python
suspects = [4, 7, 8, 17]
controls = [19, 1, 16, 12]
r3 = json.loads((W/"probe_3"/"results.json").read_text())
r4 = json.loads((W/"probe_4"/"results.json").read_text())
agent_ix = np.array([i for i in range(98) if i % 14 < 7])
def map_metrics(a):
    a = a.astype(np.float64)
    mass = a.sum(-1)
    p = a / np.maximum(mass[...,None], 1e-20)
    entropy = -(p*np.log(np.maximum(p, 1e-30))).sum(-1)/np.log(a.shape[-1])
    af = p[...,agent_ix].sum(-1)
    return mass.mean(-1), af.mean(-1), entropy.mean(-1)
m3 = np.stack([np.stack(map_metrics(np.load(r["attention"]["29"]))) for r in r3])
m4 = np.stack([np.stack(map_metrics(np.load(r["attention"]["29"]))) for r in r4])
for hs, label in [(suspects, "suspect"), (controls, "control")]:
    print(label, "L29 video mass / agent fraction / entropy")
    print(" original:", np.round(m3[:,:,hs].mean((0,2)),4).tolist())
    print(" cap4:    ", np.round(m4[:,:,hs].mean((0,2)),4).tolist())
for rows, label in [(r3,"original"),(r4,"cap4")]:
    aa = [np.load(r["action"]).reshape(-1,7) for r in rows]
    print(label, "pair RMS first10 XYZ",
          [round(float(np.sqrt(np.mean((aa[i][:10,:3]-aa[i+1][:10,:3])**2))),5)
           for i in (0,2)])

print("roll_5 artifact top:", [p.name for p in (W/"fwd"/"roll_5").iterdir()])
screen_eps = json.loads((W/"screen_camera_episodes.json").read_text())
print("screen episodes", len(screen_eps), "first", screen_eps[:3])
print("screen variants first:", str(json.loads((W/"screen_camera_variants.json").read_text())[:1])[:600])
print("Output shape:", SHAPES["mot.mixtures.action.blocks.29.self_attn.o.weight"])

def snapshot(tag, metadata):
    d = W/"candidates"/tag
    d.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i,n in enumerate(edited()):
        fn = f"tensor_{i:03d}.pt"
        torch.save(load(n), d/fn)
        manifest[n] = fn
    (d/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (d/"metadata.json").write_text(json.dumps(metadata,indent=2))

def clear_edits():
    for n in list(edited()):
        drop_edit(n)

def set_output_heads(heads, factor, layer=29):
    clear_edits()
    name = f"mot.mixtures.action.blocks.{layer}.self_attn.o.weight"
    t = load_original(name)
    for h in heads:
        t[:, h*128:(h+1)*128] *= factor
    save_edit(name, t)
    return name

set_output_heads(suspects, 0.5)
snapshot("l29_suspect_o_half", {"layer":29, "heads":suspects, "retained_output":0.5,
                               "mechanism":"Attenuate four agent-locked action readers only; original Q/K"})
print("Current edit:", edited())

ledger = json.loads((W/"pass_ledger.json").read_text())
if not any(x["pass"] == 5 for x in ledger):
    ledger.append({"pass":5,"type":"rollout","edit":"l29_qk_cap4","episodes":128,
                   "paired_diff_pp":1.5625,"se_pp":2.93,"wrong_to_right":8,"right_to_wrong":6,
                   "purpose":"Discovery screen; inconclusive"})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with notes.open("a") as f:
    f.write("\n## Passes 4-5: first intervention not established\n"
            "- cap4 discovery: 63/128 original vs 65/128 edited; +1.6 pp, SE 2.9; 8 gains/6 losses. Inconclusive.\n"
            f"- L29 suspect entropy original={m3[:,:,suspects].mean((0,2))[2]:.3f}, cap4={m4[:,:,suspects].mean((0,2))[2]:.3f}.\n"
            "- Next candidate: original Q/K; halve only L29 self-attention output columns for H4/7/8/17.\n"
            "- Current best remains original pending stronger evidence. Forward passes used 5/40.\n")
```
