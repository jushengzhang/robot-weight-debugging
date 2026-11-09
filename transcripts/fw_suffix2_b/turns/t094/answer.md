The first clean batch shows no net loss (+0.8 points, SE 3.0). I’ll save both results and prepare the first reserved paraphrase batch, which checks that the repair still follows legitimate rewordings rather than treating unfamiliar language as a tag.
```python
validation_helper = r'''
def validation_status(log=True, next_request=True):
    plan = json.loads((W/"validation_plan.json").read_text())
    groups = {"raw":[], "strip":[], "language":[]}
    completed = []
    for p in plan:
        path = W/f'roll_{p["pass"]}.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        expected = {(r["suite"],r["id"],r["seed"]) for r in p["request"]["episodes"]}
        for model in ("original","edited"):
            got = [(r["suite"],r["id"],r["seed"]) for r in rows if r["model"]==model]
            assert len(got)==len(expected) and set(got)==expected, (p["pass"],model)
        group = "language" if p["split"].startswith("language_") else p["condition"]
        groups[group].append(p["pass"])
        completed.append(p["pass"])
    report = {}
    for g,ks in groups.items():
        if ks:
            m = paired_rolls(ks)
            m["passes"] = ks
            m["ci95_pp"] = [m["diff_pp"]-1.96*m["se_pp"],m["diff_pp"]+1.96*m["se_pp"]]
            report[g] = m
    (W/"validation_status.json").write_text(json.dumps(report,indent=2))
    brief = {g:{k:v for k,v in m.items() if k!="by_category"} for g,m in report.items()}
    print(json.dumps(brief,indent=1))
    if log:
        with (W/"notes.md").open("a") as f:
            f.write("\nFrozen validation through pass"+str(max(completed))+": "+json.dumps(brief)+"\n")
    if next_request:
        pending = [p for p in plan if p["pass"] not in completed]
        if pending:
            p=pending[0]
            print("NEXT",p["pass"],p["split"],p["condition"])
            print(json.dumps(p["request"],separators=(",",":")))
    return report
'''
(W/"validation_helpers.py").write_text(validation_helper)
exec(validation_helper, globals())
status = validation_status()
```
