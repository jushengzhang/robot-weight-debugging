The matched panel again shows strong old-goal behavior: 10 of 11 task failures reach the original goal, while all semantic counterparts succeed. I will check initial-image equality and action differences now; only genuinely matched observations can support a clean instruction-sensitivity analysis.
```python
r8=read_jsonl(W/"roll_8.jsonl")
r8id={r["id"]:r for r in r8}
matched_initial=[]
exact_prompt_pairs=[]
print("Initial task vs semantic: pixel RMSE base/wrist, noise equality, action translation RMS")
for gi,k in enumerate(matched_groups):
    es=complete[k]
    z={}
    for typ,e in es.items():
        r=r8id[e["id"]]
        z[typ]=torch.load(first_input(r),map_location="cpu",weights_only=False)
        matched_initial.append({"group":gi,"ptype":typ,"id":e["id"],"path":first_input(r)})
    rms=[]
    for key in ["observation/image","observation/wrist_image"]:
        x=np.asarray(z["task"]["element"][key],np.float32)
        y=np.asarray(z["semantic"]["element"][key],np.float32)
        rms.append(float(np.sqrt(np.mean((x-y)**2))))
    noise_eq=np.array_equal(z["task"]["noise"],z["semantic"]["noise"])
    da=np.asarray(z["task"]["action_chunk"])-np.asarray(z["semantic"]["action_chunk"])
    act_rms=float(np.sqrt(np.mean(da[:,:3]**2)))
    print(gi,k[0],round(rms[0],3),round(rms[1],3),noise_eq,round(act_rms,4),
          "task_success",r8id[es["task"]["id"]]["success"])
    if max(rms)<1e-6 and noise_eq:
        exact_prompt_pairs.append(gi)
print("Exact image-matched groups:",exact_prompt_pairs)
print("Probe paths task/semantic only:")
probe9_paths=[]
for gi,k in enumerate(matched_groups):
    for typ in ["task","semantic"]:
        probe9_paths.append(f'WORK/roll_8_frames/e{complete[k][typ]["id"]}_seed7_original/t0000_inputs.pt')
print(json.dumps(probe9_paths))
(W/"matched_initial.json").write_text(json.dumps(matched_initial))
# Per-episode attention-output magnitudes: diagnostics only, trajectories differ.
print("Expert attention output norms, mean across 80 episodes:")
for l in range(18):
    values=[float(np.linalg.norm(np.load(r["activations"])[f"mean::expert.{l}.attn_out::action"])) for r in r8]
    print(l,round(float(np.median(values)),2),round(float(np.max(values)),2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 8: matched baseline80 (16 groups, all five ptypes)\n"
            "Task5/16 (10/11 failures old goal), position6/16, environment9/16, object13/16, semantic16/16. "
            f"Exact initial image-matched task/semantic groups: {exact_prompt_pairs}.\n")
```
