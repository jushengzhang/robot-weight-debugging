The L15 visual value direction separates the held-out physical-state examples well: pairwise separation is 95.4%, although the controls are not perfectly zero. I will now check how the relevant head writes that direction into gripper and arm channels. This distinguishes a missing visual cue from a cue whose downstream effect is weak or inappropriate.
```python
print("Physical cue's existing gated head write: XYZ/rotation/grip readout, averaged over first five calls")
for layer,head in ((3,2),(8,2),(15,2)):
    d=physical_dirs[layer]["value_direction"]
    oh=load_original(E+f"layers.{layer}.self_attn.o_proj.weight")[:,head*256:(head+1)*256]
    write=oh@d
    gate=torch.from_numpy(branch_data[layer]["ga"].astype(np.float32))
    eff=(gate[:5]*write)@outw[:7].T
    print(layer,head,"value-contrast scale",round(physical_dirs[layer]["raw_scale"],3),
          "write norm",round(float(write.norm()),3),
          "mean velocity-readout",np.round(eff.mean(0).numpy(),4).tolist())
    print("  z and grip first/last:",np.round(eff[[0,4]][:,[2,6]].numpy(),4).tolist())

# Prepare a conservative, single-head, rank-one motion test.
layer,head,beta=15,2,2.0
d=physical_dirs[layer]["value_direction"]
gate=torch.from_numpy(branch_data[layer]["ga"][:5].mean(0).astype(np.float32))
z=outw[2]*gate
protect=(outw[[0,1,3,4,5,6]]*gate).T
Q=torch.linalg.qr(protect,mode="reduced").Q
u=z-Q@(Q.T@z)
u=u/u.norm()
n=E+f"layers.{layer}.self_attn.o_proj.weight"
w0=load_original(n)
w=w0.clone()
# Flow integration is x <- x - dt*v, so a negative z-velocity write raises predicted z.
w[:,head*256:(head+1)*256] -= beta*torch.outer(u,d)
save_edit(n,w)
cue_spec={"name":"E15H2_empty_cue_retreat_beta2","layer":layer,"head":head,"beta":beta,
          "direction_file":"empty_cue_retreat_directions.pt","purpose":"physical empty-like cue to upward reset; preserve other direct action readouts"}
torch.save({"value_direction":d,"write_direction":u},W/"empty_cue_retreat_directions.pt")
(W/"current_candidate.json").write_text(json.dumps(cue_spec,indent=2))
print("Rank-one edit relative tensor norm",round(float((w-w0).norm()/w0.norm()),4),
      "protected readout overlap",round(float((protect.T@u).abs().max()),7))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 26: physical-state cue diagnosis\n"
            "- Correct aperture is abs(q1-q2), not q1+q2 (opposite-signed finger joints).\n"
            "- 15 fully-closed/empty-like, 16 holding-like, 16 open controls; derive/check split by scene hash.\n"
            "- L15 value direction protected against holding/open/base/language means: held-out score means 0.608 empty, 0.228 holding, 0.149 open; pairwise separation 95.4%. Cue is not perfectly specific.\n"
            "- Candidate: rank-one addition to E15H2 O block routes this visual cue toward upward reset. Other direct pose/grip readouts orthogonalized. beta=2 diagnostic, not yet evidence of recovery improvement.\n")
```
