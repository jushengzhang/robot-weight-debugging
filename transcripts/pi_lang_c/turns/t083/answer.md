The audit confirms that only 65 embedding rows differ from the checkpoint; all tested non-language prompts use untouched rows. I will now create the replication requests with fixed new seeds. These results will test F without changing it.
```python
reserved = []
for batch in range(2, 6):
    reserved.extend(json.loads((W / f"confirm_{batch}_episodes.json").read_text()))
assert len(reserved) == 460
assert len({(e["suite"], e["id"]) for e in reserved}) == 460
practice_keys = {(v["suite"], v["id"]) for v in variants}
assert all((e["suite"], e["id"]) in practice_keys for e in reserved)

for repeat, seed in enumerate([3101, 6203]):
    for j, batch in enumerate(range(2, 6)):
        pass_id = 35 + 4 * repeat + j
        eps = json.loads((W / f"confirm_{batch}_episodes.json").read_text())
        req = {"type": "rollout", "model": "both",
               "episodes": [{"suite": e["suite"], "id": e["id"], "seed": seed} for e in eps]}
        if pass_id == 35:
            req["save_frames"] = True
        (W / f"F_request_{pass_id}.json").write_text(json.dumps(req))
print("RESERVED VARIANTS BY SUITE:", dict(collections.Counter(e["suite"] for e in reserved)))
print("REQUEST35:")
print((W / "F_request_35.json").read_text())
```
