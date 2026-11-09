The early lexical edit improved task success by two episodes, but the overall gain is uncertain and the old-goal count did not fall. I will inspect those changes and test a second, stronger dose. If task following does not improve consistently with dose, I will not attribute the small gain to repaired instruction binding.
```python
r25 = read_jsonl(W/"roll_25.jsonl")
arm25 = defaultdict(dict)
for r in r25:
    arm25[r["id"]][r["model"]] = r
for eid, a in arm25.items():
    o,e = a["original"],a["edited"]
    if o["success"] != e["success"] or (o["ptype"]=="task" and o["orig_goal_reached"] != e["orig_goal_reached"]):
        print(eid, o["ptype"], "success", o["success"], "->", e["success"],
              "oldgoal", o["orig_goal_reached"], "->",e["orig_goal_reached"],
              "instruction:",o["instruction"],
              "moved_edit:",{k:v for k,v in e["obj_moved_m"].items() if v>0.05})

recipe = torch.load(W/"content_embedding_gain150_recipe.pt", weights_only=True)
E = load_original(emb_name)
E[recipe["ids"]] = recipe["original_rows"] + (recipe["original_rows"]-recipe["function_mean"])
save_edit(emb_name, E)
del E
print("Installed lexical contrast gain2.0, changed rows:",len(recipe["ids"]))
with notes.open("a") as f:
    f.write(
        "\n## Pass25: lexical contrast gain1.5\n"
        "Dev69->70/128, +0.8 +/-2.1pp, 4 wins/3 losses. Env17->16, object23 unchanged, "
        "position3->2, semantic21->22, task5->7. Old-goal failures13 unchanged. "
        "Testing dose2.0 on the same 103 rows before judging the mechanism.\n"
    )
```
