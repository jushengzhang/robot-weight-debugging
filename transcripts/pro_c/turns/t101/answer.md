The completed validation rejects the final candidate: it achieved 369/640 successes versus 379/640 for the original model. I will remove the edit and verify the submission state. Leaving a measured regression in place would contradict the precommitted decision rule.
```python
def paired_summary(pass_numbers):
    pairs = defaultdict(dict)
    for k in pass_numbers:
        for line in (W / f"roll_{k}.jsonl").read_text().splitlines():
            r = json.loads(line)
            pairs[(r["id"],r["seed"])][r["model"]] = r
    assert all(set(v) == {"original","edited"} for v in pairs.values())
    def summarize(items):
        a = np.array([int(v["original"]["success"]) for v in items], dtype=float)
        b = np.array([int(v["edited"]["success"]) for v in items], dtype=float)
        d = b-a
        return {"n":len(d), "original_successes":int(a.sum()), "edited_successes":int(b.sum()),
                "gain_pp":float(100*d.mean()),
                "paired_se_pp":float(100*d.std(ddof=1)/np.sqrt(len(d))),
                "wrong_to_right":int((d>0).sum()), "right_to_wrong":int((d<0).sum())}
    vals = list(pairs.values())
    result = {"overall":summarize(vals),
              "by_ptype":{p:summarize([v for v in vals if v["original"]["ptype"]==p])
                          for p in sorted({v["original"]["ptype"] for v in vals})}}
    return result, pairs

j_final, j_pairs = paired_summary(range(36,41))
d_final, _ = paired_summary(range(26,31))
expected_ids = {e["id"] for e in json.loads((W / "J_validation_640.json").read_text())}
assert {k[0] for k in j_pairs} == expected_ids and len(j_pairs) == 640
retain = (j_final["overall"]["gain_pp"] > 2*j_final["overall"]["paired_se_pp"]
          and j_final["by_ptype"]["position"]["gain_pp"] > 0
          and all(j_final["by_ptype"][p]["gain_pp"] >= -2 for p in ["object","semantic"]))
assert not retain

removed = list(edited())
for name in removed:
    drop_edit(name)
assert not list(edited())
assert torch.equal(load(Jname), load_original(Jname))
assert torch.equal(load(Oname), load_original(Oname))
report = {"final_choice":"original_weights_no_edits",
          "active_edits":list(edited()), "J_validation":j_final,
          "D_validation":d_final,
          "conclusion":"No tested edit demonstrated a reliable independent robustness gain. All edits removed."}
(W / "final_report.json").write_text(json.dumps(report, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Final decision after40 passes\n")
    f.write("J0.75 independent640: " + json.dumps(j_final) + "\n")
    f.write("Rejected: no overall gain and position benefit did not replicate. "
            "D0.5 also failed its independent640 validation (+0.47pp,SE~0.95). "
            "ALL EDITS REMOVED. Final submission uses original weights; edited() is empty. "
            "No robustness improvement claimed. This avoids introducing measured regressions "
            "and preserves original performance by weight identity. Full summary: final_report.json.\n")
print("J independent validation:", json.dumps(j_final, indent=2))
print("D independent validation overall:", d_final["overall"])
print("Removed:", removed)
print("FINAL ACTIVE EDITS:", list(edited()))
print("Final expert tensors exactly match originals: verified.")
```
