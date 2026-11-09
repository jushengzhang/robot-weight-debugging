The saved inputs use bfloat16 tensors, which caused the comparison helper to fail before completing the audit. I will convert those tensors to float32 for analysis, finish the replay check, and remove the rejected edit.
```python
def asnp(x):
    return x.detach().float().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x,dtype=np.float32)
def rms_np(a,b):
    return float(np.sqrt(np.mean((asnp(a)-asnp(b))**2)))
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
print("Remaining edits:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass16: video-value edit not accepted\n"
            "- Paired diagnostics: original BG50/65 vs edited46/65; light35/37 vs34/37. Overall1 rescue,6 regressions, gain -4.9pp SE2.6.\n"
            "- Original baseline differs from earlier same-seed originals (which had BG48/65); auditing saved initial observations/actions.\n"
            "- Video edits removed. Passes used16/40; no accepted edit.\n")
```