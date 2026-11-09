Validation is complete. The projection reduced tagged success by 2.54 percentage points, so I will remove it rather than submit a known regression. This finalization step will save the pooled results and verify that the submitted weights are the originals.
```python
# Finalization only: no further forward passes or candidate search.
removed = list(edited())
for tensor_name in removed:
    drop_edit(tensor_name)
assert len(edited()) == 0
assert torch.equal(
    load("mot.mixtures.video.text_embedding.0.weight"),
    load_original("mot.mixtures.video.text_embedding.0.weight")
)

for p in range(40, 51):
    verify_rollout(p)
_ = log_validation(50)

validation = {
    "tagged": paired_rollout([40, 41, 42, 43]),
    "clean": paired_rollout([44, 45, 46]),
    "paraphrase": paired_rollout([47, 48, 49, 50])
}

def read_condition(passnos, model):
    result = {}
    for p in passnos:
        for line in (W / f"roll_{p}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["model"] == model:
                key = (row["suite"], row["id"], row["seed"])
                assert key not in result
                result[key] = row
    return result

suffix_penalty = {}
for model in ("original", "edited"):
    raw = read_condition([40, 41, 42], model)
    clean = read_condition([44, 45, 46], model)
    assert raw.keys() == clean.keys()
    d = np.array([
        int(clean[k]["success"]) - int(raw[k]["success"]) for k in raw
    ], dtype=float)
    suffix_penalty[model] = {
        "n": len(d),
        "raw_success": sum(x["success"] for x in raw.values()),
        "clean_success": sum(x["success"] for x in clean.values()),
        "clean_minus_raw_pp": 100 * float(d.mean()),
        "paired_se_pp": 100 * float(d.std(ddof=1) / np.sqrt(len(d)))
    }

report = {
    "forward_passes_used": 50,
    "final_choice": "original checkpoint; all edits removed",
    "submitted_edits": list(edited()),
    "rejected_candidate": "frozen_video_orthogonal_rank6",
    "candidate_tensor": "mot.mixtures.video.text_embedding.0.weight",
    "candidate_transform": "W_new = W_original - W_original @ Q6 @ Q6.T",
    "independent_validation": validation,
    "matched_suffix_penalty": suffix_penalty,
    "decision_reason": (
        "No validated tagged-success gain. Final projection lost 13 successes "
        "in 512 independent paired tagged episodes. Earlier gate also failed "
        "independent validation. Restore original rather than retain an "
        "unsupported or harmful intervention."
    )
}
(W / "final_report.json").write_text(json.dumps(report, indent=2))
with (W / "notes.md").open("a") as f:
    f.write(
        "\n## FINAL SUBMISSION AFTER PASS50\n"
        "ALL EDITS REMOVED; original checkpoint submitted. Verified edited()==[] "
        "and former edited tensor equals load_original exactly.\n"
        "Frozen rank6 independent results:\n"
        + json.dumps(validation, indent=2)
        + "\nMatched suffix penalty:\n"
        + json.dumps(suffix_penalty, indent=2)
        + "\nNo candidate achieved a validated tagged gain while satisfying safety. "
        "The requested robust repair was not achieved. Full final summary: final_report.json.\n"
    )

print("REMOVED:", removed)
print("FINAL ACTIVE EDITS:", list(edited()))
print("VALIDATION:", json.dumps(validation, indent=2))
print("MATCHED SUFFIX PENALTY:", json.dumps(suffix_penalty))
```
