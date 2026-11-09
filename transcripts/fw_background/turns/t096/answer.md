I’ll compare the action changes with the signed effect predicted by the edited readout, including the gripper channel. If the effect is texture-selective and agrees with that prediction, I’ll move to paired rollouts; reduced action discrepancy is still not evidence of better task success.
```python
p20 = probe_results(20)
assert [r["input"] for r in p19] == [r["input"] for r in p20]
A0 = np.stack([np.load(r["action"]) for r in p19])
A1 = np.stack([np.load(r["action"]) for r in p20])
AN0 = np.stack([np.load(r["action_norm"]) for r in p19])
AN1 = np.stack([np.load(r["action_norm"]) for r in p20])

def matched_differences(A):
    return np.concatenate([A[T[:,0]]-A[T[:,2]], A[T[:,1]]-A[T[:,2]]],axis=0)

D0,D1 = matched_differences(A0),matched_differences(A1)
predicted = dose*(Xraw@u).numpy()[:,None]*(H0@u).numpy()[None,:]
observed = (AN1-AN0).mean(1)
pred_cos = float(np.sum(predicted*observed)/
                 (np.linalg.norm(predicted)*np.linalg.norm(observed)+1e-15))
summary20 = {
    "pass":20,"dose":dose,
    "bg_action_change_rms":rms(A1[:54]-A0[:54]),
    "light_action_change_rms":rms(A1[54:]-A0[54:]),
    "original_bg_light_discrepancy_rms":rms(D0),
    "edited_bg_light_discrepancy_rms":rms(D1),
    "bg_light_discrepancy_ratio":rms(D1)/rms(D0),
    "original_mean_bg_light_bias":D0.mean((0,1)).tolist(),
    "edited_mean_bg_light_bias":D1.mean((0,1)).tolist(),
    "predicted_observed_change_cosine":pred_cos,
    "predicted_change_rms":rms(predicted),
    "observed_normalized_mean_change_rms":rms(observed),
    "physical_change_by_channel":np.sqrt(np.mean((A1[:54]-A0[:54])**2,axis=(0,1))).tolist(),
    "light_change_by_channel":np.sqrt(np.mean((A1[54:]-A0[54:])**2,axis=(0,1))).tolist()
}
print(json.dumps(summary20,indent=2))
(W / "probe20_summary.json").write_text(json.dumps(summary20,indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass20: rank1 readout causal probe\n")
    f.write("- "+json.dumps(summary20)+"\n")

# Save reusable, original-referenced setter for subsequent dose and negative-control tests.
utility = '''
import json
import numpy as np
import torch
from pathlib import Path

def set_readout_edit(dose, kind="target"):
    W = Path(WORK)
    for name in list(edited()):
        drop_edit(name)
    z = np.load(W / "readout_causal_directions.npz")
    direction = torch.from_numpy(z[kind].copy())
    name = "mot.mixtures.action.head.weight"
    original = load_original(name)
    modified = original-dose*torch.outer(original@direction,direction)
    if dose:
        save_edit(name,modified)
    spec = {"family":"task_protected_texture_mean_readout",
            "dose":dose,"kind":kind,"tensor":name,
            "relative_change":float((modified-original).norm()/original.norm())}
    (W / "current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
'''
(W / "experiment_utils.py").write_text(utility)
exec(utility)
print("Active:", json.loads((W / "current_candidate.json").read_text())["family"])
```
