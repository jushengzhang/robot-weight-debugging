The backoff repairs most of the earlier collateral failures, but the dairy category prototype may be too weak to recover cheese descriptions. I will inspect the per-case outcomes and use the saved probes to distinguish a wrong dairy sense from a more general need for context-sensitive features.
```python
r17=read_roll(17); p17=paired_records(r17)
print("D targeted test:",paired_summary(17))
c_lookup={(r["suite"],r["id"]):r for k in [14,15,16] for r in read_roll(k)
          if r["model"]=="edited" and r.get("termination") in ("success","max_steps")}
for p in p17.values():
    a,b=p["original"],p["edited"]; c=c_lookup.get((a["suite"],a["id"]))
    print(a["suite"],a["id"],"orig/C/D",int(a["success"]),int(c["success"]) if c else "-",int(b["success"]),a["instruction"])
# Inspect source dairy tokenization and broad cue occurrences.
for text in ["spreadable dairy product", "cultured milk", "cold dairy liquid", "dairy spread",
             "churned milk fat", "cream cheese", "merlot"]:
    ids=sp.encode(text)
    print("TOK",text,list(zip(ids,sp.id_to_piece(ids))))
with (W/"notes.md").open("a") as f:
    f.write("\n## D targeted test (pass 17)\n24 hard selected cases, original6->edited13, 8WR/1RW. "
            "Need inspect retained dairy regression and lost cheese recoveries; class centroid alone "
            "may not encode the distinguishing modifier. All results are development, not confirmation.\n")
```
