The combined edit gained only one validation success, so it did not meet the predeclared selection threshold. I will freeze the smaller five-row, 50%-residual edit and evaluate it unchanged on the reserved confirmation batches. If those independent results do not support it, I will restore the original weights rather than submit a development-set win.
```python
r31 = read_jsonl(W/"roll_31.jsonl")
counts31 = Counter(r["ptype"] for r in r31 if r["model"]=="edited" and r["success"])
success31 = sum(r["success"] for r in r31 if r["model"]=="edited")
choose_combined = success31 >= 85 and counts31["object"] >= 23 and counts31["semantic"] >= 22
assert not choose_combined
for n in list(edited()):
    drop_edit(n)
save_edit(pos_name, loaded_candidate)
frozen_tensor = load(pos_name)
frozen_hash = hashlib.sha256(frozen_tensor.numpy().tobytes()).hexdigest()
torch.save({pos_name:frozen_tensor}, W/"frozen_candidate.pt")
(W/"frozen_candidate.json").write_text(json.dumps({
    "candidate":"five exceptional positional rows, local residual retain50%",
    "name":pos_name, "changed_rows":torch.where((frozen_tensor!=P).any(-1))[0].tolist(),
    "tensor_sha256":frozen_hash, "frozen_after_pass":31,
    "confirmation_batches":[f"final_batch_{i}.json" for i in range(1,7)],
    "safety_batches":[f"safety_batch_{i}.json" for i in range(1,3)],
}, indent=2))

def paired_summary(pass_numbers, only_ptype=None):
    pairs = {}
    for k in pass_numbers:
        for r in read_jsonl(W/f"roll_{k}.jsonl"):
            if only_ptype and r["ptype"] not in only_ptype:
                continue
            key=(r["id"],r["seed"])
            pairs.setdefault(key,{})[r["model"]]=r
    pairs = [p for p in pairs.values() if "original" in p and "edited" in p]
    d=np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in pairs],float)
    return {
        "n":len(d),
        "original_successes":sum(p["original"]["success"] for p in pairs),
        "edited_successes":sum(p["edited"]["success"] for p in pairs),
        "diff_pp":round(100*float(d.mean()),3) if len(d) else None,
        "se_pp":round(100*float(d.std(ddof=1)/np.sqrt(len(d))),3) if len(d)>1 else None,
        "wins":int((d>0).sum()),"losses":int((d<0).sum())
    }

with notes.open("a") as f:
    f.write(
        "\n## Pass31 and freeze\n"
        "Combined edit validation82->83/128, +0.8 +/-2.6pp, 6 wins/5 losses. "
        "Env13->14, object24->23, position11->10, semantic22->24, task12 unchanged. "
        "Does not meet predeclared selection threshold. Freeze sparse five-row keep50% "
        "exactly as pass12/19. Frozen hash "+frozen_hash+". "
        "No further tuning; six untouched final batches plus two safety batches next.\n"
    )
print("Frozen changed rows:", torch.where((frozen_tensor!=P).any(-1))[0].tolist())
print("Frozen hash:",frozen_hash)
print("Final batch1:", (W/"final_batch_1.json").read_text())
```
