The late-MLP target has not shown a meaningful, selective action effect. I will compare it with the control and, if that confirms the weak result, remove all edits. I will then freeze the checkpoint and use the remaining seven passes for paired rollback checks on the preselected independent variants—one category per pass. Those checks will not be presented as robustness gains.
```python
s33 = summarize_late_probe(33, 1.0, "control")
print("Norm-matched control:", json.dumps({
    k:v for k,v in s33.items()
    if k not in ["failure_action_change_by_channel","light_action_change_by_channel","failure_pair_ratios"]
}, indent=1))
print("Target vs control discrepancy reduction (percentage of discrepancy):",
      round(100*(1-s32["failure_control_discrepancy_ratio"]),4),
      round(100*(1-s33["failure_control_discrepancy_ratio"]),4))

removed_names = list(edited())
for name in removed_names:
    drop_edit(name)
assert list(edited()) == []
assert all(torch.equal(load(n), load_original(n)) for n in removed_names)
print("Rollback verified for last six tensors; edited() is empty.")

diag_eps = readj("diagnostic_episodes.json")
assert isinstance(diag_eps,list)
diag_ids = {(e["suite"],int(e["id"])) for e in diag_eps}
audit_manifests = {cat:batches[0] for cat,batches in vm.items()}
for cat, es in audit_manifests.items():
    assert len(es)==128 and len({(e["suite"],e["id"]) for e in es})==128
    assert not (diag_ids & {(e["suite"],int(e["id"])) for e in es})
(W/"final_rollback_audit_manifests.json").write_text(json.dumps(audit_manifests,indent=2))
(W/"current_candidate.json").write_text(json.dumps({
    "family":"original_checkpoint", "retained_edits":[],
    "decision":"No tested intervention demonstrated a supported background improvement.",
    "remaining_passes":"Seven category-wise paired rollback audits; no further selection."
},indent=2))
with (W/"notes.md").open("a") as f:
    f.write(
        "\n## Pass33 and frozen final choice\n"
        "- Orthogonal late-MLP control: "+json.dumps(s33)+"\n"
        "- Target doses .5/1.0 erase the recorded feature but do not produce a meaningful "
        "selective action correction. Reject late MLP candidate; all edits removed.\n"
        "- No successful weight repair was found. Freeze original checkpoint rather than retain a "
        "harmful or unsupported edit. Final seven passes use the preselected first128 independent "
        "practice variants of each category as paired rollback audits. These are no-op checks, "
        "not evidence of improved robustness. No further parameter selection.\n"
    )

print("\nNext request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Background Textures"]},separators=(",",":")))
```
