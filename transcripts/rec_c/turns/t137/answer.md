The confirmation test rejects the lift-only repair: 3 wins and 9 losses, or -4.8 ± 2.7 percentage points. I will remove it. One remaining hypothesis is that a visibly empty closed grasp needs both reopening and modest clearance; I will test that feature-gated action change rather than retain the harmful lift edit.
```python
for name in list(edited()):
    drop_edit(name)
r45 = [json.loads(x) for x in (W/"roll_45.jsonl").read_text().splitlines() if x.strip()]
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass45: H1-only clearance0.60 rejected on untouched confirmation\n"
            "126 pairs,63 vs69 successes,3 wins/9 losses (-4.8+/-2.7pp). "
            "Loop failures40 vs41, total failures63 vs57: reduced loop count is not a recovery gain. "
            "Removed. No validated repair.\n")
vd = designs["clearance_value_edit_design.pt"]
C = vd["readout_constraints"].double()
vnom = vd["value_direction_per_nominal"].double()
target = vd["target"].double()
print("constraint target reshaped",target.reshape(8,7).tolist())
Cv = C@vnom
scale = float((Cv@target)/(target@target))
print("C @ stored per-nominal direction scale:",scale,
      "residual",float((Cv-scale*target).norm()))
print("constraint rank",int(torch.linalg.matrix_rank(C)),
      "condition",float(torch.linalg.cond(C)))
print("prior route per nominal norm",float(vnom.norm()))
# Inspect only saved input metadata, not a model execution.
saved_input = torch.load(Path(m[0]["source"]), map_location="cpu", weights_only=False)
print("Saved input top keys",list(saved_input) if isinstance(saved_input,dict) else type(saved_input).__name__)
print("Contact examples from geometry manifest:")
cm = json.loads((W/"contact_manifest.json").read_text())
print(json.dumps(cm[:1] if isinstance(cm,list) else list(cm)[:3],indent=1)[:1600])
```
