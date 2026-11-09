The layer-11 projection did not improve discovery success, so I will revert it. The weak action effects suggest that raw attention-output magnitude is a poor guide here. I will inspect post-block residuals next to find where viewpoint differences enter the representation that reaches the action readout.
```python
clear_edits()
print("Action non-block tensors:")
for n in NAMES:
    if "mot.mixtures.action." in n and ".blocks." not in n:
        print(n, SHAPES[n])
print("Layer-0 modulation/FFN tensor names:")
for n in NAMES:
    if "mot.mixtures.action.blocks.0." in n and ("modulation" in n or "ffn" in n):
        print(n, SHAPES[n])
mods = [n for n in NAMES if "mot.mixtures.action.blocks." in n and n.endswith(".modulation")]
if mods:
    for l in range(30):
        n = f"mot.mixtures.action.blocks.{l}.modulation"
        if n in NAMES:
            t=load_original(n)
            if l in [0,5,10,11,16,20,25,29]:
                print("L",l,"modulation RMS per slot:",np.round(t.reshape(-1,1024).square().mean(1).sqrt().numpy(),4).tolist())
ledger = json.loads((W/"pass_ledger.json").read_text())
ledger.append({"pass":9,"type":"rollout","edit":"l11_rank1_center_half","episodes":128,
               "paired_diff_pp":-1.5625,"se_pp":3.33,"wrong_to_right":8,"right_to_wrong":10,
               "decision":"rejected"})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with notes.open("a") as f:
    f.write("\n## Passes 8-9: L11 rank-one output projection rejected\n"
            "- Stable camera direction (split-task cosine .921); analytic rank-one 50% output projection, mean-light bias preserved.\n"
            "- Local coefficient halved as predicted, but first10 XYZ camera-pair RMS only .071218 -> .070788 (-0.6%). L16 shift unchanged.\n"
            "- Discovery rollout: 63/128 original vs 61/128 edited; -1.6 pp, SE 3.3. Reverted.\n"
            "- Need diagnose post-block residuals/gating; large raw attention shifts are not necessarily behaviorally influential.\n"
            "- Forward passes used 9/40. Current best ORIGINAL, active edit NONE.\n")
print("Active edits:",edited())
```
