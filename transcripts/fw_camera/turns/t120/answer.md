Objects Layout is unchanged overall. I’ll use one of the last three passes to independently replicate the Sensor Noise safety flag, pooling both batches regardless of the result. This reduces the planned camera sample size, but neither the weights nor the retention thresholds will change; the final two passes will remain fixed camera panels.
```python
for k in (35,36,37):
    log_frozen_rollout(k,f"Safety: {safety_categories[k-32]}")
safety_all={safety_categories[k-32]:paired_summary(paired_rows([k])) for k in range(32,38)}
print("Safety results:")
for cat,s in safety_all.items():
    print(cat,s)
print("All noncamera pooled:",paired_summary(paired_rows(range(32,38))))
(root/"safety_validation_progress.json").write_text(json.dumps(safety_all,indent=2))

# One additional, disjoint Sensor Noise batch. No outcome-based variant selection.
seen_sensor={(p["original"]["suite"],p["original"]["id"]) for p in paired_rows([36])}
sensor_pool=[v for v in practice if v["category"]=="Sensor Noise" and (v["suite"],v["id"]) not in seen_sensor]
sensor_pool.sort(key=lambda v:hashlib.sha256(
    f'sensor-safety-replication-v1:{v["suite"]}:{v["id"]}'.encode()).hexdigest())
sensor_confirm=[{"suite":v["suite"],"id":v["id"],"seed":230000+i} for i,v in enumerate(sensor_pool[:128])]
assert len(sensor_confirm)==128
old_req38=json.loads((root/"request_38.json").read_text())
(root/"request_38_camera_unused.json").write_text(json.dumps(old_req38))
req38={"type":"rollout","model":"both","episodes":sensor_confirm}
(root/"request_38.json").write_text(json.dumps(req38))
plan=json.loads((root/"final_evaluation_plan.json").read_text())
plan["amendment_after_pass37"]={
    "reason":"Sensor Noise first batch -2.344pp +/-2.593pp SE: replicate possible safety loss",
    "pass38":"128 disjoint Sensor Noise variants selected by hash; pool passes36+38",
    "pass39_40":"unchanged preselected camera panels",
    "weights_and_retention_thresholds":"unchanged",
    "final_camera_episodes":1242,
    "no_additional_safety_retesting_after_pass38":True
}
(root/"final_evaluation_plan.json").write_text(json.dumps(plan,indent=2))
with (root/"notes.md").open("a") as f:
    f.write("\n## Final-budget amendment after pass37\n"
            "- All6 categories checked at128 paired episodes each: Light0pp, Robot0pp, Language-1.563pp, Background+.781pp, Sensor-2.344pp, Layout0pp.\n"
            "- Sensor safety flag exceeds2pp point-estimate tolerance but SE~2.6pp. Allocate pass38 to one disjoint128-variant Sensor replication; pool36+38 regardless of result. No further safety retesting.\n"
            "- Passes39-40 remain their preselected camera panels; final camera n1242 instead of1370. All weights and original retention thresholds unchanged.\n"
            "- Camera gain through31 still unconfirmed (+1.420pp, clusterSE1.100); original remains fallback.\n")
print("NEXT REQUEST:",json.dumps(req38,separators=(",",":")))
```
