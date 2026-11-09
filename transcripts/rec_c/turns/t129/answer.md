Centering did not eliminate control-state changes: the edit adds about 0.11 vertical action units on both contact and open controls. I will inspect the two heads separately across denoising steps before using fresh confirmation scenes. Their first-step feature separation may not hold through the full action-generation process.
```python
for k in [35,36,41,42,43]:
    rr = json.loads((W/f"probe_{k}"/"results.json").read_text())
    r = next(r for r in rr if r["arm"]=="orig")
    if "activations" in r:
        z = np.load(r["activations"])
        print("probe",k, {x:list(z[x].shape) for x in z.files})
gatefile = np.load(W/"final_attention_gates.npz")
print("gates", {k:list(gatefile[k].shape) for k in gatefile.files})
print("reference nonzeros", torch.nonzero(cd["reference"]).flatten().tolist(),
      cd["reference"][cd["reference"]!=0].tolist())
parts = cd["parts"]
C = designs["clearance_value_edit_design.pt"]["readout_constraints"]
for h, p in parts.items():
    r = (C@p).reshape(8,7)
    print("part",h, "physical response Z by head", [round(float(x),5) for x in r[:,2]],
          "other physical RMS", float(r[:,[0,1,3,4,5,6]].square().mean().sqrt()))
print("manifest sequence", [(r["index"],group_key(r)) for r in m[:44]])
```
