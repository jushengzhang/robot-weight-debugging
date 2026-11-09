Single-head query softening is closer to neutral than the previous edits: two rescues, three losses, with gains concentrated in L2 adaptation scenes. I will inspect those changes and test a milder, pre-specified gain of 0.75; this is still discovery, not evidence of a general improvement.
```python
r32=read_roll(32)
p32=defaultdict(dict)
for r in r32:p32[(r["id"],r["grip"],r["seed"])][r["model"]]=r
for k,d in p32.items():
    a,b=d["original"],d["edited"]
    if a["success"]!=b["success"]:
        print(k,a["level"],a["instruction"],"success",a["success"],b["success"],
              "flips",a["grip_flips"],b["grip_flips"],"steps",a["steps"],b["steps"])
qw=load_original(qname); qw[768:1024]*=0.75; save_edit(qname,qw)
(W/"current_candidate.json").write_text(json.dumps({"name":"E16H3_Q_0.75","layer":16,"head":3,"query_gain":0.75}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 32: E16H3 query gain 0.5 screen\n"
            "- 31/64 -> 30/64, -1.6 pp (SE 3.5), 2 rescues / 3 losses; loop failures 21 -> 20.\n"
            "- L2 +12.5 pp, L1 -12.5 pp, L3 -6.2 pp; not evidence of an overall gain.\n"
            "- Next same-head dose: query gain 0.75, chosen before its outcomes.\n")
```
