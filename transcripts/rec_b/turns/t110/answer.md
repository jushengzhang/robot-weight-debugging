The initial arm poses are not identical in every pair, so I will exclude pairs with appreciable pose changes and check that their object-jitter settings match. This avoids labeling object motion or arm motion as a “gripper feature.” The remaining contrast will still be approximate because closing fingers can affect nearby objects.
```python
br={(r["id"],r["grip"]):r for r in base_rows}
print("Object jitter example:",br[(38,"open")]["obj_deltas"])
print("Same-pair closed jitter:",br[(38,"close")]["obj_deltas"])

eligible=[]
for p in initial_pairs:
    sid=p["id"]
    jo=json.dumps(br[sid,"open"]["obj_deltas"],sort_keys=True)
    jc=json.dumps(br[sid,"close"]["obj_deltas"],sort_keys=True)
    if p["pose_delta"]<1e-4 and jo==jc:
        eligible.append(p)
print("Eligible matched pose+jitter pairs:",len(eligible))
eligible=sorted(eligible,key=lambda p:(p["pose_delta"],p["id"]))[:16]
grip_pair_ids=[p["id"] for p in eligible]
print("Selected IDs:",grip_pair_ids)
print("Levels:",dict(Counter(scene_by_id[i]["level"] for i in grip_pair_ids)))

grip_req={"type":"probe","model":"original",
          "inputs":[f"WORK/roll_1_frames/s{i}_{g}_seed17_original/t0000_inputs.pt"
                    for i in grip_pair_ids for g in ["open","close"]],
          "return":{"layers":[],"vlm_layers":[1,4,6,13,16],
                    "vision_layers":[26],"what":"resid"}}
print(json.dumps(grip_req,indent=1))
(W/"gripper_image_probe_request.json").write_text(json.dumps(grip_req))
```
