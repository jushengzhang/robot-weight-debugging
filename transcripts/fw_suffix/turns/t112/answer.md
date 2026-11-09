The final tagged estimate is +1.8 points with a paired SE of about 1.4 points: not convincing, and accompanied by a 3.1-point paraphrase loss. I need one cleanup execution before the final summary to remove the rejected candidate and ensure the submitted weights are actually restored.
```python
final_results = {
    "tagged": paired_summary([34,35,40]),
    "clean_strip": paired_summary([36,37]),
    "language_instructions": paired_summary([38,39])
}
removed = list(edited())
for n in removed:
    drop_edit(n)
assert len(edited()) == 0
remaining = [p for p in (W/"edited").rglob("*")
             if p.is_file() and p.suffix in (".npy", ".pt", ".safetensors")]
assert not remaining, "Unexpected tensor files remain in the submission directory"

final_report = {
    "final_choice": "original_weights_restored",
    "submitted_edited_tensors": 0,
    "removed_tensor_edits": len(removed),
    "rejected_candidate": json.loads((W/"frozen_candidate.json").read_text()),
    "independent_validation": final_results,
    "reason": (
        "No tested edit met the retention criteria. The frozen gate's tagged gain "
        "was not clearly larger than its paired uncertainty, and Language Instructions "
        "success dropped by 3.125 percentage points. No successful tag-invariance "
        "repair is claimed."
    ),
    "forward_passes_used": 40
}
(W/"final_report.json").write_text(json.dumps(final_report,indent=2))
(W/"current_edit.json").write_text(json.dumps({
    "name": "none_original_restored", "edited_count": 0,
    "final_report": "WORK/final_report.json"
},indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## FINAL CHOICE AFTER PASS40: ORIGINAL WEIGHTS RESTORED\n"
            "All308 frozen-gate edits removed; edited() is empty and no tensor files "
            "remain in WORK/edited. No instrumentation remains. No successful repair "
            "found under the stated constraints.\n"
            +json.dumps(final_results)+"\n"
            "Tag gain was not statistically convincing; paraphrase loss exceeded "
            "the approximately2pp tolerance. Full audit in final_report.json; rejected "
            "candidate snapshot remains outside the submission directory in "
            "fwd/probe_33_weights.\n")
print(json.dumps(final_results,indent=2))
print("FINAL SUBMISSION: original weights; edited tensors =",len(edited()))
```
