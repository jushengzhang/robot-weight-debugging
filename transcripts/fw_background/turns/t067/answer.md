The narrowed action-output edit did not improve diagnostic success: background fell by 3.1 points and light by 2.7 points, with substantial paired uncertainty. I will reject it. The next test targets the observation side instead, because suppressing action-output directions can remove useful task responses along with texture effects.
```python
clear_edits()
def paired_report(pass_id):
    rows=[json.loads(s) for s in (W/f"roll_{pass_id}.jsonl").read_text().splitlines()]
    groups={}
    for r in rows:
        groups.setdefault((r["suite"],r["id"],r["seed"]),{})[r["model"]]=r
    report={}
    for cat in sorted(set(r["category"] for r in rows)):
        pairs_=[v for v in groups.values() if "original" in v and "edited" in v
                and v["original"]["category"]==cat]
        d=np.array([int(v["edited"]["success"])-int(v["original"]["success"]) for v in pairs_])
        report[cat]={"n":len(d),"gain_pp":float(d.mean()*100),
                     "se_pp":float(d.std(ddof=1)/np.sqrt(len(d))*100) if len(d)>1 else 0.,
                     "rescues":int((d==1).sum()),"regressions":int((d==-1).sum())}
    return report
r13stats=paired_report(13)
print("Rejected rollout result:",r13stats)
(W/"roll13_summary.json").write_text(json.dumps(r13stats,indent=2))
print("Forward artifact layout (names only):")
for sub in [W/"fwd/roll_2", W/"fwd/roll_2/artifacts"]:
    print(str(sub.relative_to(W)), [p.name for p in sub.iterdir()][:12])
print("Video means available in original diagnostic:")
z=np.load(r2[0]["activations"])
print([(k,z[k].shape) for k in z.files if "video.0." in k])
z.close()
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 13: action-output projection rejected by rollouts\n"
            "- Paired diagnostic 102 episodes. BG original48/65 vs edited46/65; light35/37 vs34/37. Overall4 rescues,7 regressions, gain -2.9pp SE3.3pp.\n"
            "- All action-output edits removed. Probe invariance alone was insufficient; shrinking downstream task-response directions can hurt policy execution.\n"
            "- Next localization targets video observation features before shared K/V transfer. Passes used13/40; no accepted edit.\n")
```