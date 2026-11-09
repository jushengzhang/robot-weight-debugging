The first rollout test shows one repair and no regressions, but its uncertainty is too large to retain the edit on that result alone. I will inspect the null terminations and then test full-strength row replacement to distinguish weak semantic correction from a failure elsewhere in the policy.
```python
r7=read_roll(7)
print("n",len(r7),"first",json.dumps(r7[0],ensure_ascii=True)[:1500])
pair7=collections.defaultdict(dict)
for r in r7:
    pair7[r["suite"],r["id"],r["seed"]][r["model"]]=r
print("models",collections.Counter(r["model"] for r in r7))
for key,rr in pair7.items():
    if len(rr)==2:
        o,e=rr["original"],rr["edited"]
        if o.get("termination") is None or e.get("termination") is None or o["success"]!=e["success"]:
            print(key,"success",o["success"],e["success"],"term",o.get("termination"),e.get("termination"),"extra", {k:v for k,v in e.items() if k in ["error","error_type","instruction"]})
set_embedding_pairs(pairs_small,1.0,"lex_small_full")
print("full strength edit saved")
with (W/"notes.md").open("a") as f:
    f.write("\nPass 6 causal replay, alpha=.5: cooktop action L2 distance to stove control 2.409 -> .123; "
            "wine description distance 1.120 -> .639; all 8 unrelated controls bit-identical. "
            "Paired first observations have identical images and flow noise. "
            "Pass 7 alpha=.5: +1.6 +/-1.6 pp (1 repair, 0 regressions) on 64 goal d4-5, "
            "but 13 null terminations need accounting. Testing full strength next.\n")
```
