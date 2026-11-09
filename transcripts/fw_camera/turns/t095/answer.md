I’ll measure the replay changes against the original and earlier edit, then prepare the same discovery rollout panel used before. The replay test checks the proposed mechanism; the rollout test will determine whether that translates into task success.
```python
def get_probe_actions(k, arm, field="action", n=57):
    return np.stack([np.load(root/f"probe_{k}"/f"probe_{arm}_{i}_{field}.npy") for i in range(n)])

def replay_metrics(k, arm="edit"):
    out={}
    for field in ("action","action_norm"):
        base=get_probe_actions(2,"orig",field).reshape(19,3,32,7)
        cur=get_probe_actions(k,arm,field).reshape(19,3,32,7)
        bo=np.sqrt(np.mean((base[:,1,:10,:3]-base[:,0,:10,:3])**2,axis=(1,2)))
        ce=np.sqrt(np.mean((cur[:,1,:10,:3]-cur[:,0,:10,:3])**2,axis=(1,2)))
        change=np.sqrt(np.mean((cur[:,:,:10,:3]-base[:,:,:10,:3])**2,axis=(0,2,3)))
        out[field]={"original_pair_xyz_rms":float(bo.mean()),
                    "edited_pair_xyz_rms":float(ce.mean()),
                    "relative_pct":float(100*(ce.mean()/bo.mean()-1)),
                    "pairs_improved":int((ce<bo).sum()),
                    "xyz_edit_rms_mild_severe_light":change.tolist(),
                    "gripper_sign_disagreement":np.mean(
                        np.sign(cur[:,:,:10,6])!=np.sign(base[:,:,:10,6]),axis=(0,2)).tolist()}
    return out

print("Probe24 result sample:", str(json.loads((root/"probe_24/results.json").read_text()))[:160])
# Tool uses 'edit' artifact prefix for the edited arm.
prefix24=next((root/"probe_24").glob("*_0_action.npy")).name.split("_")[1]
m24=replay_metrics(24,prefix24)
print("Revised half replay:",json.dumps(m24))
print("Earlier half replay:",json.dumps(replay_metrics(16,next((root/"probe_16").glob("*_0_action.npy")).name.split("_")[1]))[:650])
(root/"probe24_analysis.json").write_text(json.dumps(m24,indent=2))
notes=(root/"notes.md").read_text().replace(
    "Full rank79 control protection erased camera-action alignment;",
    "Full rank79 control protection weakened camera-action alignment (cos .5005 vs revised .6034);")
(root/"notes.md").write_text(notes)
screen_eps=json.loads((root/"screen_camera_episodes.json").read_text())
if isinstance(screen_eps,dict):
    print("Screen keys:",list(screen_eps))
    screen_eps=screen_eps["episodes"]
print("Discovery episode count:",len(screen_eps))
req25={"type":"rollout","model":"both","episodes":screen_eps[:128]}
(root/"request_25.json").write_text(json.dumps(req25))
print("NEXT REQUEST:",json.dumps(req25,separators=(",",":")))
```
