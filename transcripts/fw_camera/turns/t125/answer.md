Pass 39 is slightly negative, and the pooled camera benefit remains unconfirmed. I will record the aggregate and retrieve the last preselected panel without changing either the candidate or the retention rule.
```python
from collections import Counter

camera_passes = [20, 21, 22, 23, 28, 29, 30, 31, 39]
camera_rows = read_pairs(camera_passes)
assert len(camera_rows) == 1114
assert all(r["category"] == "Camera Viewpoints" for r in camera_rows)
assert len({(r["suite"], r["id"], r["seed"]) for r in camera_rows}) == len(camera_rows)

for run in [28, 29, 30, 31, 39]:
    observed = Counter((r["suite"], r["id"], r["seed"])
                       for r in camera_rows if r["run"] == run)
    expected = Counter((r["suite"], r["id"], r["seed"])
                       for r in plan["camera_panels"][str(run)])
    assert observed == expected, run

s39 = paired_summary(read_pairs([39]))
camera_summary = paired_summary(camera_rows)
camera_summary["ci95_cluster_pp"] = [
    camera_summary["diff_pp"] - 1.96 * camera_summary["cluster_se_pp"],
    camera_summary["diff_pp"] + 1.96 * camera_summary["cluster_se_pp"]
]
print("Pass39:", rounded(s39))
print("Independent camera aggregate:", rounded(camera_summary))
print("Current camera retention condition:",
      camera_summary["diff_pp"] > 1.96 * camera_summary["cluster_se_pp"])

(W / "camera_validation_progress.json").write_text(
    json.dumps({"passes": camera_passes, "summary": camera_summary}, indent=2)
)
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass39: penultimate fixed camera seed panel\n")
    f.write("- Frozen candidate unchanged. " + json.dumps(rounded(s39)) + "\n")
    f.write("- Independent camera aggregate: " +
            json.dumps(rounded(camera_summary)) + "\n")
    f.write("- Camera retention criterion still not met; finish fixed pass40. "
            "No tuning, no selective omission of any validation panel.\n")

request40 = {"type": "rollout", "model": "both",
             "episodes": plan["camera_panels"]["40"]}
assert len(request40["episodes"]) == 128
assert hashlib.sha256(load("mot.mixtures.action.head.weight").contiguous().numpy().tobytes()).hexdigest() == lock["tensor_sha256"]
(W / "next_request.json").write_text(json.dumps(request40, separators=(",", ":")))
print("NEXT REQUEST:", json.dumps(request40, separators=(",", ":")))
```
