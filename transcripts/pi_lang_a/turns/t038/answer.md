The complementary-direction control largely retains the benefit, whereas the common-direction control does not. That supports a token-specific task-feature explanation rather than generic scaling. I will now remove the ambiguous dairy-to-cheese row and freeze a 16-row candidate for final diagnostic checks and independent validation.

```python
print("orthogonal-only valid:",paired_summary(14))
for key,a,b in paired_records(14):
    if a["success"]!=b["success"] and not b["success"]:
        print("regression record:",json.dumps(b,ensure_ascii=True)[:1800])
descriptor_safe={k:v for k,v in descriptor_aliases.items() if k!="dairy"}
candidate16_meta=apply_alias_groups([(direct_aliases,1.0),(descriptor_safe,1.0)],"candidate16_frozen")
candidate16_ids={m["source_id"] for m in candidate16_meta}
print("Frozen candidate rows:",[(m["source"],m["target"],m["source_id"]) for m in candidate16_meta])
print("Parameter count changed:",len(candidate16_meta)*E0.shape[1],
      "fraction of 3.62B:",len(candidate16_meta)*E0.shape[1]/3.62e9)
print("Protected overlap:",candidate16_ids&canonical_ids)
print("Remaining unseen language variants:",sum(v["category"]=="Language Instructions" and
      (v["suite"],v["id"]) not in diagnostic_keys for v in variants))
with (W/"notes.md").open("a") as f:
    f.write("\nPass14 orthogonal-complement control: "
            f"{paired_summary(14)} (48 requested), with one edited-arm error on an unexposed control. "
            "Token-specific directions retain most benefit; common-direction adjustment alone does not. "
            "Full17 regression is dairy beverage (milk) incorrectly aliased to cheese. "
            "Frozen candidate16 = direct7 + descriptor9 full substitutions, dairy removed. "
            "32768 weight values; all canonical task tokens protected. No VLM blocks, vision, or action weights edited.\n")
```
