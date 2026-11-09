The final batch also regressed, so I will not submit this candidate. I will clear the edits, verify that the original tensor is restored, and record the pooled results before giving the final summary.
```python
p49 = paired_rows(49)
p50 = paired_rows(50)
fresh_final = fresh47 + p49 + p50
final_confirmation = p49 + p50
unsolved_final = [p for p in fresh_final
                  if not (p[1].get("success_at_t0",False) or p[2].get("success_at_t0",False))]
report = {
    "decision":"REJECT all tested edits; submit original checkpoint",
    "reset_candidate_all_fresh":pair_summary(fresh_final),
    "reset_candidate_final_confirmation_only":pair_summary(final_confirmation),
    "reset_candidate_excluding_t0_solved":pair_summary(unsolved_final),
    "prefail_proxy_only_not_ordinary_benchmark":pair_summary(proxy47),
    "by_level":{level:pair_summary([p for p in fresh_final if p[1]["level"]==level])
                for level in ["L1","L2","L3","L4"]},
    "limitation":"A causally controllable visual-to-gripper route was found, but no validated recovery repair."
}
removed = list(edited())
for name in removed:
    drop_edit(name)
assert not edited(), "Submission still contains edits"
assert torch.equal(load(vname),load_original(vname)), "Original tensor not restored"
report["submitted_edits"] = edited()
report["restoration_verified"] = True
(W/"final_report.json").write_text(json.dumps(report,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## FINAL DECISION after all50 forward passes\n"
            "No edits retained. Final reset intervention failed independent validation.\n"
            +json.dumps(report,indent=2)+"\n"
            "All submission edits removed; load(vname) exactly equals load_original(vname). "
            "Candidate archives remain outside edited/ for audit only. "
            "No claim of improved recovery or successful localization of a unique loop cause.\n")
print("All fresh recovery pairs:",report["reset_candidate_all_fresh"])
print("Final confirmation only:",report["reset_candidate_final_confirmation_only"])
print("Excluding already-solved starts:",report["reset_candidate_excluding_t0_solved"])
print("Submitted edits:",edited())
print("Original tensor restoration verified:",report["restoration_verified"])
```
