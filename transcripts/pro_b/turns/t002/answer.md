The practice list contains all five perturbation types but no explicit unperturbed episodes. I’ll group cases by scene and initial state so that comparisons do not confuse scene differences with perturbation effects, and set aside a hash-based validation partition that I will not use to construct edits.

```python
ep_by_id = {e["id"]: e for e in episodes}
def group_key(e):
    return (e["suite"], e["task"], e["init_idx"])
def split_name(e):
    s = "|".join(map(str, group_key(e)))
    h = int(hashlib.sha256(("repair-v1|" + s).encode()).hexdigest()[:8], 16) % 10
    return "diagnostic" if h < 4 else ("validation" if h < 7 else "confirmation")

partitions = {s: [e for e in episodes if split_name(e) == s]
              for s in ("diagnostic", "validation", "confirmation")}
print("partitions:", {s: len(es) for s, es in partitions.items()})
print("instruction_changed by ptype:",
      {p: dict(collections.Counter(e["instruction_changed"] for e in episodes if e["ptype"] == p))
       for p in sorted({e["ptype"] for e in episodes})})
groups = collections.defaultdict(dict)
for e in partitions["diagnostic"]:
    groups[group_key(e)][e["ptype"]] = e
complete_groups = [g for g, es in groups.items() if len(es) == 5]
print("fully matched diagnostic groups:", len(complete_groups))
rng = random.Random(104)
diag_groups = []
for suite in sorted({e["suite"] for e in episodes}):
    gs = [g for g in complete_groups if g[0] == suite]
    rng.shuffle(gs)
    seen = set()
    for g in gs:
        if g[1] not in seen:
            diag_groups.append(g); seen.add(g[1])
        if len(seen) == 4:
            break
diag_eps = [groups[g][p] for g in diag_groups
            for p in ("object", "position", "semantic", "task", "environment")]
print("diagnostic episodes:", len(diag_eps))
for g in diag_groups:
    es = groups[g]
    print(g[0], g[2], "ids", [es[p]["id"] for p in ("object","position","semantic","task","environment")])
    print(" scene:", g[1])
    print(" task instruction:", es["task"]["instruction"])
diag_req = {"type":"rollout", "model":"original",
            "episodes":[{"id":e["id"],"seed":17} for e in diag_eps],
            "record":{"vlm_layers":[0,3,6,9,12,15,17],
                      "expert_layers":[0,3,6,9,12,15,17],
                      "vision_layers":[0,8,17,26],"what":"resid"},
            "save_frames":True}
(W / "diagnostic_request.json").write_text(json.dumps(diag_req))
(W / "split_manifest.json").write_text(json.dumps(
    {s:[e["id"] for e in es] for s,es in partitions.items()}))
with (W / "notes.md").open("a") as f:
    f.write("\n## Data split\nGrouped by (suite, task, init_idx), SHA256 repair-v1: "
            "40% diagnostic, 30% validation, 30% confirmation. "
            "No unperturbed ptype is present in the supplied episode list.\n")
print("\nREQUEST:")
print(json.dumps(diag_req))
```
