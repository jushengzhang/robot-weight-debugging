The action mean is almost exactly explained by the final residual readout, and the old protection space greatly weakened alignment with the camera-induced action error. I’ll test a revised rank-two edit that preserves the action encoder, the clean mean, and the leading clean-task modes covering 85% of their variance. This retains reproducible camera directions while avoiding the overly restrictive full control span.
```python
import shutil

# A priori spectral rule: smallest power-of-two clean rank explaining >=85%
# of centered control variance, after protecting the direct action encoder.
clean_rank = next(r for r in (1,2,4,8,16,32,56)
                  if float(sc[:r].square().sum()/sc.square().sum()) >= 0.85)
Qnew, Unew, _ = basis_options[clean_rank]
b0 = load_original("mot.mixtures.action.head.bias").double()
oldU = torch.from_numpy(np.load(root / "readout_camera_U.npy")).double()
if oldU.shape[0] != 1024:
    oldU = oldU.T
for label, u in [("old_full_control_protection", oldU), ("new_dominant_control_protection", Unew)]:
    perr=(D@u)@(W0@u).T
    print(label, "camera-error cosine",
          round(float(torch.nn.functional.cosine_similarity(perr.flatten(),delta_linear.flatten(),dim=0)),4),
          "predicted half-dose mean-action discrepancy ratio",
          round(float(torch.linalg.norm(delta_linear-0.5*perr)/torch.linalg.norm(delta_linear)),4))

def install_centered_camera(alpha, u=Unew, label=None):
    for name in list(edited()):
        drop_edit(name)
    dw = (W0@u)@u.T
    we = (W0-alpha*dw).float()
    be = (b0+alpha*(dw@clean_mean)).float()
    save_edit(wname,we)
    save_edit("mot.mixtures.action.head.bias",be)
    if label:
        dest = root/"candidates"/label
        shutil.copytree(root/"edited",dest,dirs_exist_ok=True)
        (dest/"metadata.json").write_text(json.dumps({
            "alpha":alpha,"rank":int(u.shape[1]),"clean_rank":clean_rank,
            "mean_preserved":True,"source":"Matched-state discovery residuals only"
        },indent=2))
    return we,be

new_half_label = "readout_camera_rank2_clean16_center_half"
we,be = install_centered_camera(0.5,label=new_half_label)
np.savez(root/"revised_camera_subspace.npz",
         U=Unew.numpy(),protect=Qnew.numpy(),clean_mean=clean_mean.numpy(),
         clean_singular=sc.numpy(),clean_rank=clean_rank)
print("New edit:",new_half_label)
print("Protected rank:",Qnew.shape[1], "clean variance retained:",
      round(float(sc[:clean_rank].square().sum()/sc.square().sum()),4))
print("Relative head-weight change:",float(torch.linalg.norm(we.double()-W0)/torch.linalg.norm(W0)))
print("Protected mapping error:",float(torch.linalg.norm((we.double()-W0)@Qnew)))
print("Clean mean mapping error:",float(torch.linalg.norm((we.double()-W0)@clean_mean+be.double()-b0)))
print("Bias change norm:",float(torch.linalg.norm(be.double()-b0)))

replay_inputs = json.loads((root/"compact_replay_inputs.json").read_text())["original"]
req24={"type":"probe","model":"edited","inputs":replay_inputs,
       "return":{"actions":True,"layers":[29],"what":"resid","noise_seed":46001}}
(root/"request_24.json").write_text(json.dumps(req24))
with (root/"notes.md").open("a") as f:
    f.write("\n## Revised mechanism after pass23 (no validation-derived directions)\n"
            "- Mean32 action contrast equals negative final residual readout (cos .99990).\n"
            "- Full rank79 control protection erased camera-action alignment; revised analytic rule preserves encoder+bias and top clean PCs covering85% energy (rank16; total protected24).\n"
            "- Revised nuisance rank2 reproduces across task split (principal cos .689/.656), camera-readout-error cosine .603, half-dose predicted mean-action discrepancy ratio .874.\n"
            "- Mean clean readout preserved by compensating head bias; only head.weight and head.bias edited.\n"
            "- Active candidate readout_camera_rank2_clean16_center_half, not yet tested. Original remains fallback.\n"
            "- Reserved 128 never-run camera variants in final_camera_reserve.json. Planned remaining passes: half/full replay+discovery, negative control, frozen validation, all six safety categories.\n")
print("Probe input count:",len(replay_inputs),"request saved")
print(json.dumps(req24,separators=(",",":")))
```
