Four categories showed no outcome differences, and the camera difference was an execution error. Light Conditions remains the unresolved safety check. I will prepare a saved-input audit with its three discordant episodes plus 125 previously untested severe-light variants, and freeze the next language evaluations in advance.
```python
# New light stress set: exclude the first audit; repeat its 3 discordances only for diagnosis.
old_light={(r["suite"],r["id"]) for r in read_roll(27)}
light_pool=[r for r in practice if r["category"]=="Light Conditions"
            and r["difficulty"] in (4,5) and (r["suite"],r["id"]) not in old_light]
print("Available disjoint severe-light variants:",len(light_pool))
rng=np.random.default_rng(203)
new_light=[light_pool[i] for i in rng.choice(len(light_pool),125,replace=False)]
light_repeats=[{"suite":p["original"]["suite"],"id":p["original"]["id"],"seed":p["original"]["seed"]}
               for p in pair_roll(27) if bool(p["original"]["success"])!=bool(p["edited"]["success"])]
light_req={"type":"rollout","model":"both",
           "episodes":light_repeats+[{"suite":r["suite"],"id":r["id"],"seed":37} for r in new_light],
           "save_frames":True}
(W/"light_stress_request.json").write_text(json.dumps(light_req,indent=1))
val37=[{"type":"rollout","model":"both",
        "episodes":[{"suite":r["suite"],"id":r["id"],"seed":37} for r in batch]}
       for batch in manifest]
(W/"validation_seed37_requests.json").write_text(json.dumps(val37,indent=1))
seen_lang={(r["suite"],r["id"]) for p in W.glob("roll_*.jsonl")
           for r in [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
           if r["category"]=="Language Instructions"}
fresh_language=[r for r in practice if r["category"]=="Language Instructions"
                and (r["suite"],r["id"]) not in seen_lang]
(W/"final_fresh_language_manifest.json").write_text(json.dumps(fresh_language,indent=1))
print("Still entirely untested practice-language variants:",len(fresh_language))
print("First safety audit (all pairs):")
for k in range(25,31):
    ps=pair_roll(k)
    print(k,ps[0]["original"]["category"],paired_stats(ps))
with (W/"notes.md").open("a") as f:
    f.write("\nPasses25-30 first safety audit, 128/category: Camera88->87 (only error regression); "
            "Robot94->94; Light117->114 (3 true rollout discordances); Background120->120; "
            "Sensor110->110; Layout109->109. No audit prompt contains any edited row. "
            "Light has tiny first-action differences on14/128 despite no touched tokens; prepare saved-input audit "
            "on3 discordances +125 disjoint difficulty4-5 variants, explicit rollout seed37. "
            "Important: sample.seed selects variants; sample rollouts use seed7. Explicit episodes set actual rollout seed.\n"
            "Plan passes31-50: light saved-input stress; seed37 validation4x128; untouched remaining language; "
            "causal semantic-direction/norm controls and destination ablation; other-category replication; "
            "additional language seed/stress tests; final error accounting/replay. Frozen chosen edit stays lex_dest_exact.\n")
print("\nLIGHT STRESS REQUEST:")
print(json.dumps(light_req,separators=(",",":")))
```
