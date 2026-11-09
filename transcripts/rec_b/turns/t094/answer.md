Both directions of the positional edit hurt recovery, so I will restore those embeddings. The next hypothesis is different: current visual changes may be weakly transmitted through the VLM value projections, even when attention reaches the images. I will compare within-task visual variation with instruction and camera-common components, without fitting any weights to success labels.
```python
for n in list(edited()):
    drop_edit(n)
sel=json.loads((W/"mlp_probe_selection.json").read_text())
late_sel=[s for s in sel if s["step"]>0]
print("Late matched observations:",len(late_sel))
print("Example alias files:",[p.name for p in (W/"roll_1_frames"/"s10_close_seed17_original").iterdir()][:6])

# Summarize source-specific linear input/output overlap; no network execution.
ain=load_original("action_in_proj.weight")
aout=load_original("action_out_proj.weight")
cosin=(ain.T@ain)/(ain.norm(dim=0)[:,None]*ain.norm(dim=0)[None,:])
print("Gripper input cosine with pose columns:",[round(float(v),3) for v in cosin[6,:6]])
print("Action input column norms (pose,grip,padding mean):",
      [round(float(x),2) for x in ain.norm(dim=0)[:7]],round(float(ain[:,7:].norm(dim=0).mean()),2))

probe_req={"type":"probe","model":"original",
           "inputs":[s["input"] for s in late_sel],
           "return":{"layers":[],"vlm_layers":list(range(17)),
                     "vision_layers":[0,5,10,15,20,26],"what":"resid"}}
print(json.dumps(probe_req,indent=1))
(W/"visual_content_probe_request.json").write_text(json.dumps(probe_req))
with (W/"notes.md").open("a") as f:
    f.write("\n## Positional-register amplification rejected, pass36\n"
            "Residual x1.5:57/96->54/96,-3.1+/-2.8pp,2 rescues/5 regressions. "
            "Both positional doses harm L4; original registers retained. "
            "Current edits NONE. Next: measure transmission of within-task visual "
            "variation through VLM value projections, separate from common/camera/language directions.\n")
```
