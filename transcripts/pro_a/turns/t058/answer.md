The semantic-subspace gain did not improve task success and increased regrasp failures, so I will remove it. I will next test the visual-feedback pathway instead: layer 14 heads 0 and 1 attend strongly to wrist-image locations that differ between approach and grasp stages, unlike the nearly fixed hotspot in layer 9. Strengthening these two paths may help the controller respond to the current grasp rather than repeat a familiar trajectory.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass21: L12 semantic-subspace gain1.5\n"
        "Dev69->68/128, -0.8 +/-2.1pp, 3 wins/4 losses. "
        "Env17->15, object23->22, position3 and task5 unchanged, semantic21->23. "
        "Old-goal failures13->12, loops29->35. REJECT current semantic-value gain.\n"
    )
for n in list(edited()):
    drop_edit(n)

o14_name = f"{expert_root}.14.self_attn.o_proj.weight"
O14 = load_original(o14_name)
O14edit = O14.clone()
O14edit[:, :512] *= 1.5  # Only heads 0 and 1; other 6 heads unchanged.
save_edit(o14_name, O14edit)
torch.save({o14_name: O14edit}, W/"wrist_feedback_L14_H01_gain150.pt")
print("Installed visual-feedback candidate:", edited())
print("Per-head original output norms:", [round(float(O14[:,256*h:256*(h+1)].norm()),2) for h in range(8)])
print("Relative matrix change:", round(float((O14edit-O14).norm()/O14.norm()),4))

# Quantify how fixed initial-target failures and regrasp failures differ.
for pt in ["position", "environment", "task", "object"]:
    rs = [r for r in r1 if r["ptype"] == pt]
    for success in [False, True]:
        ss = [r for r in rs if bool(r["success"]) == success]
        if not ss:
            continue
        print(pt, "success", success, "n", len(ss),
              "median_flips", round(float(np.median([r["grip_flips"] for r in ss])),1),
              "median_eef_path", round(float(np.median([r["eef_path_m"] for r in ss])),2),
              "median_total_object_move",
              round(float(np.median([sum(r["obj_moved_m"].values()) for r in ss])),2))
with notes.open("a") as f:
    f.write(
        "\nCandidate I: expert L14H0/H1 o_proj columns0:512 x1.5; all other weights original. "
        "Probe18: wrist attention initial tasks64-81%, grasp-stage position/object81-99%; "
        "top locations differ (w193/w114 initial vs w254/w242 grasp). "
        "Hypothesis: increase current visual feedback for grasp correction; not gripper/action scaling. "
        "Stage comparisons are not pure intervention evidence; test rollouts and matched-stage maps next.\n"
    )
```
