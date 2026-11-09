The video-value candidate also failed the paired rollout test, so I will not keep it on the strength of probe invariance. The reported original baseline changed from earlier runs on the same episode list; I will check saved initial inputs and actions to distinguish simulator variability from a possible baseline-replay problem before interpreting further dose tests.
```python
r16=[json.loads(s) for s in (W/"roll_16.jsonl").read_text().splitlines()]
r16lookup={(r["suite"],r["id"],r["model"]):r for r in r16}
p15=read_probe(15)
p15lookup={obskey(r["input"])[:2]:r for r in p15}
def load_initial(r):
    return torch.load(Path(r["frames_dir"])/"t0000_inputs.pt",map_location="cpu",weights_only=False)
def rms_np(a,b):
    return float(np.sqrt(np.mean((np.asarray(a,dtype=np.float32)-np.asarray(b,dtype=np.float32))**2)))
print("suite/id: original input differences and initial-action replay checks")
for suite,vid in [("libero_object",1),("libero_spatial",207),("libero_object",179),
                  ("libero_spatial",180),("libero_10",2266),("libero_goal",2317)]:
    old=load_initial(rlookup[(suite,vid)])
    ro=load_initial(r16lookup[(suite,vid,"original")])
    re_=load_initial(r16lookup[(suite,vid,"edited")])
    pe=np.load(p15lookup[(suite,vid)]["action"])
    print(suite,vid,{
        "old_new_image":round(rms_np(old["input_image"],ro["input_image"]),7),
        "old_new_proprio":round(rms_np(old["proprio"],ro["proprio"]),7),
        "old_new_original_action":round(rms_np(old["action_chunk"],ro["action_chunk"]),7),
        "original_vs_edited_initial":round(rms_np(ro["action_chunk"],re_["action_chunk"]),7),
        "probe_edit_vs_new_original":round(rms_np(pe,ro["action_chunk"]),7),
        "probe_edit_vs_new_edited":round(rms_np(pe,re_["action_chunk"]),7),
        "seeds":[old["seed"],ro["seed"],re_["seed"]]
    })
print("Paired stats:",paired_report(16))
(W/"roll16_summary.json").write_text(json.dumps(paired_report(16),indent=2))
clear_edits()
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass16: video-value edit not accepted\n"
            "- Paired diagnostics: original BG50/65 vs edited46/65; light35/37 vs34/37. Overall1 rescue,6 regressions, gain -4.9pp SE2.6.\n"
            "- Original baseline differs from earlier same-seed originals (which had BG48/65), so inspect exact saved observations/actions before further causal interpretation.\n"
            "- Video edits removed. Passes used16/40; no accepted edit.\n")
```