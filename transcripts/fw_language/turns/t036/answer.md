The larger baseline succeeds on 90/128 episodes, with failures concentrated in goal and long-horizon tasks; all 28 object-suite episodes succeed. I will select natural success/failure instruction pairs for the same tasks and verify that their initial pixels and proprioception are exactly equal before probing them.
```python
import shutil
shutil.rmtree(W/"contrast_inputs")
(W/"contrast_manifest.json").unlink(missing_ok=True)
roll3=read_jsonl(W/"roll_3.jsonl")
diagnostic_rows=roll1+roll3
seen_keys={(r["suite"],r["id"]) for r in diagnostic_rows}
groups=collections.defaultdict(list)
for r in diagnostic_rows:
    groups[(r["suite"],task_of(r),r["seed"])].append(r)

def lexical_overlap(r):
    base=re.sub(r"^[A-Z_]+SCENE\d+_","",task_of(r)).replace("_"," ").lower()
    a=set(re.findall(r"[a-z]+",base))
    b=set(re.findall(r"[a-z]+",r["instruction"].lower()))
    return len(a&b)/max(len(a),1)

natural_pairs=[]
input_cache={}
def first_input(r):
    key=(r["suite"],r["id"],r["seed"])
    if key not in input_cache:
        input_cache[key]=torch.load(Path(r["frames_dir"])/"t0000_inputs.pt",
                                   map_location="cpu",weights_only=False)
    return input_cache[key]

for g,rows in groups.items():
    good=[r for r in rows if r["success"]]
    bad=[r for r in rows if not r["success"]]
    if not good or not bad: continue
    good=max(good,key=lexical_overlap)
    bad=min(bad,key=lexical_overlap)
    a,b=first_input(good),first_input(bad)
    same=torch.equal(a["input_image"],b["input_image"]) and torch.equal(a["proprio"],b["proprio"])
    ar=np.asarray(a["action_chunk"])
    br=np.asarray(b["action_chunk"])
    natural_pairs.append({"good":good,"bad":bad,"same_inputs":same,
                          "action_rms":float(np.sqrt(np.mean((ar-br)**2)))})
print("Tasks with success/failure language pairs:",len(natural_pairs))
print("Exactly equal initial image+proprio:",sum(p["same_inputs"] for p in natural_pairs))
for p in sorted(natural_pairs,key=lambda p:p["action_rms"],reverse=True)[:12]:
    print(p["bad"]["suite"],"good",p["good"]["id"],"bad",p["bad"]["id"],
          "same",p["same_inputs"],"action RMS",round(p["action_rms"],3))
    print(" good:",p["good"]["instruction"])
    print(" bad: ",p["bad"]["instruction"])

# Eight distinct task pairs, emphasizing initial action divergence without reusing tasks.
selected_pairs=sorted([p for p in natural_pairs if p["same_inputs"]],
                      key=lambda p:p["action_rms"],reverse=True)[:8]
probe_paths=[]
for p in selected_pairs:
    for side in ("good","bad"):
        r=p[side]
        probe_paths.append("WORK/"+str((Path(r["frames_dir"])/"t0000_inputs.pt").relative_to(W)))
print("\nProbe paths:",json.dumps(probe_paths,separators=(",",":")))
(W/"natural_probe_pairs.json").write_text(json.dumps(selected_pairs,indent=2))
(W/"diagnostic_seen_variants.json").write_text(json.dumps(sorted(seen_keys)))
with (W/"notes.md").open("a") as f:
    f.write("\n## Probe restriction and pass 3\n"
            "Constructed instruction-pair probe refused without charge: tool requires "
            "unmodified rollout inputs. Deleted constructed copies; no attempt to bypass. "
            "Use natural language-variant pairs with identical initial image/proprio.\n"
            "Pass 3 original language, n=128 seed2713: 90/128=70.31%; "
            "long21/38, goal15/29, object28/28, spatial26/33; diff5=10/24. "
            "These and pass1 are discovery only. 3/40 used, no edits.\n")
```
