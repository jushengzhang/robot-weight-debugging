I will predefine a matched negative-control intervention using ordinary heads in the same final layer, with output-projection norms close to the four anomalous heads. This will help test whether any improvement is specific to the suspected pathway rather than generic damping. I will then wait for backend startup; no empirical claim or weight change is justified yet.```python
suspect_heads=[(29,h) for h in (4,7,8,17)]
available=set(range(24))-{h for l,h in suspect_heads}
control_heads=[]
for l,h in suspect_heads:
    target=float(hs["action_o_head_norm"][l,h])
    candidates=[j for j in available if qk_all["frobenius"][l,j] < 70]
    j=min(candidates,key=lambda j:abs(float(hs["action_o_head_norm"][l,j])-target))
    control_heads.append((l,j))
    available.remove(j)
print("Prospective same-layer negative control:")
for (_,h),(_,j) in zip(suspect_heads,control_heads):
    print("suspect H%02d vs control H%02d: output norms %.4f / %.4f; qk coupling %.1f / %.1f" %
          (h,j,hs["action_o_head_norm"][29,h],hs["action_o_head_norm"][29,j],
           qk_all["frobenius"][29,h],qk_all["frobenius"][29,j]))
candidate_recipes={
    "status":"hypotheses only; require observation evidence before use",
    "suspect_output_heads":suspect_heads,
    "matched_control_output_heads":control_heads,
    "prospective_output_doses":[0.25,0.5,0.75,1.0],
    "alternative":"Selective capping/removal of spatial-channel excess gain, if attention maps support it",
    "selection_rule":"Prefer the smallest targeted edit with reproducible paired camera gain and no control harm"
}
(W/"prospective_interventions.json").write_text(json.dumps(candidate_recipes,indent=2))
with open(W/"notes.md","a") as f:
    f.write("\n- Complete weight scan: L29 heads 4/7/8/17 have action-query/video-key "
            "coupling Frobenius proxies 1067/900/1058/1048 vs all-head median 37; "
            "top-four singular directions account for >99.8% of their gained coupling.\n"
            "- Same-layer output-norm-matched negative-control heads predefined: "
            +str(control_heads)+". No intervention applied.\n")
print("Backend waiting interval: 600 seconds. Forward passes used: 0/40.")
time.sleep(600)
print("Wait complete; edits still:",list(edited()))
```
