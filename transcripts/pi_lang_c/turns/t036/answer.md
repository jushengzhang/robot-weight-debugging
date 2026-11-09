The first independent test does not justify keeping the 73-row candidate: its gain is only 1.6 points with a 2.9-point standard error, and six regressions offset eight recoveries. I will inspect those regressions and ablate the verb extension; this confirmation set will now be treated as development data, while the remaining reserved sets stay untouched.
```python
pairs16=paired_records(read_roll(16))
print("Independent confirmation 1:",paired_summary(16))
for p in pairs16.values():
    a,b=p["original"],p["edited"]
    if a["success"]!=b["success"]:
        ids=sp.encode(a["instruction"]+"\n")
        hits=[(sp.id_to_piece(i),sp.id_to_piece(rows_C[i]),strengths_C[i],
               "verb" if i in verb_rows else "noun/description") for i in ids if i in rows_C]
        print("FIX" if b["success"] else "BREAK",a["suite"],a["id"],"d"+str(a["difficulty"]),
              a["instruction"],"EDITS",hits)
with (W/"notes.md").open("a") as f:
    f.write("\n## Confirmation 1 (pass 16): C NOT CONFIRMED\n"
            "128 entirely new variants: 104->106, +1.56 pp SE2.93, 8 recoveries/6 regressions. "
            "Object suite 23->20. This set is now development data; remaining 460 reserved variants "
            "are still untouched. Need ablate verb extension and examine ambiguous descriptive cues.\n")
print("Regressed episode JSON:",json.dumps([{"suite":p["original"]["suite"],"id":p["original"]["id"],"seed":p["original"]["seed"]}
                                           for p in pairs16.values() if p["original"]["success"] and not p["edited"]["success"]]))
```
