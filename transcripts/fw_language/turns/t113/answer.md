The first lighting batch has 2 rescues and no regressions. I will record that result and proceed to the second disjoint lighting batch; safety on lighting does not substitute for a language improvement.
```python
def record_safety_and_prepare_next(pass_number):
    index = pass_number - 27
    expected_batch = safety_plan[index]
    pairs = paired_records([pass_number])
    expected_keys = {(e["suite"], e["id"], e["seed"]) for e in expected_batch["episodes"]}
    assert set(pairs) == expected_keys
    summary = paired_summary(pairs)
    ledger = update_evaluation_ledger()
    category = expected_batch["category"]
    print("Completed:", category, "batch", expected_batch["batch"], summary)
    print("Category cumulative:", ledger["safety"][category])
    with (work / "notes.md").open("a") as f:
        f.write(f"\n## Pass {pass_number}: safety {category}, batch {expected_batch['batch']}\n"
                "- " + json.dumps(summary) + "\n"
                "- Candidate unchanged. Full cumulative results in locked_evaluation_results.json.\n")
    if index+1 < len(safety_plan):
        print_safety_request(index+1)
    else:
        print("All twelve safety batches complete.")
    return ledger

ledger = record_safety_and_prepare_next(27)
```
