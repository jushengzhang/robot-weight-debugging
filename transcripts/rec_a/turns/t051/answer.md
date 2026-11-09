Attenuating this visual direction hurt recovery: 31/64 successes fell to 25/64, with eight losses and two rescues. I will reject it. The direction appears useful, so I will test the opposite intervention—a 50% increase along the same protected direction—to see whether stronger visual feedback helps.
```python
def apply_visual_value_edit(alpha, layers=range(1,15)):
    for n in list(edited()):
        drop_edit(n)
    for layer in layers:
        n = E + f"layers.{layer}.self_attn.o_proj.weight"
        w0 = load_original(n)
        w = w0.clone()
        d = centered_dirs[layer]
        for h in range(8):
            sl = slice(h*256,(h+1)*256)
            w[:,sl] -= alpha*torch.outer(w0[:,sl] @ d,d)
        save_edit(n,w)
    spec = {"name":f"visual_grip_value_alpha{alpha}", "layers":list(layers),
            "alpha":alpha, "directions":"visual_gripper_value_dirs_centered.pt"}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))

r17 = read_roll(17)
pair17 = {}
for r in r17:
    pair17.setdefault((r["id"],r["grip"],r["seed"]),{})[r["model"]]=r
print("Pass 17 changes: id, grip, outcome, flips original->edited")
for key,d in sorted(pair17.items()):
    o,e=d["original"],d["edited"]
    if o["success"]!=e["success"]:
        print(key[0],key[1],"rescue" if e["success"] else "loss",o["grip_flips"],e["grip_flips"])
apply_visual_value_edit(-0.5)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 17: visual-gripper value attenuation rejected\n"
            "- Fixed paired screen: 31/64 -> 25/64, -9.4 pp (SE 4.8), 2 rescues / 8 losses; loop failures 21 -> 23.\n"
            "- This visual feedback is useful, not an expendable shortcut. Reverted attenuation.\n"
            "- Next causal dose: gain 1.5 on the SAME protected value directions (alpha=-0.5), L1-14 expert O projections.\n")
print("Active candidate:",(W/"current_candidate.json").read_text())
```
