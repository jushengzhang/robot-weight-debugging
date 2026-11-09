Across all 512 fresh tagged episodes, success fell from 249 to 236. The projection therefore fails the efficacy criterion and will not be submitted. I will keep it active only to complete the planned safety measurements, starting with matched clean-instruction rollouts.
```python
print("Verified records:", verify_rollout(43))
_, tagged_final = log_validation(43)
assert tagged_final["ci95_pp"][0] <= 0
decision = {
    "candidate": "frozen_video_orthogonal_rank6",
    "retain": False,
    "reason": "Independent tagged efficacy criterion failed",
    "tagged_validation": tagged_final,
    "active_only_for_remaining_safety_measurements": True
}
(W / "final_rank6_efficacy_decision.json").write_text(json.dumps(decision, indent=2))
with (W / "notes.md").open("a") as f:
    f.write(
        "\n### Final rank6 efficacy decision after pass43\n"
        "REJECTED for submission. Independent tagged512:249->236,12 gains/25 losses. "
        "Projection remains active ONLY for preplanned clean/paraphrase characterization "
        "(passes44-50); restore original afterward. No replacement candidate will be selected "
        "from validation outcomes.\n"
    )
print("PASS44 REQUEST:")
print_request(44)
```
