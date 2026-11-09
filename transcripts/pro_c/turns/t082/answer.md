I will separate late-layer attention into image, instruction-word, BOS, and action-token mass. This will test whether failed late-stage motions are unusually dominated by action self-attention rather than the current observation.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH) / "paligemma_tokenizer.model"))

def local_path(p):
    s = str(p)
    return W / s.split("/work/", 1)[1] if "/work/" in s else Path(s)

rows1 = [json.loads(s) for s in (W / "roll_1.jsonl").read_text().splitlines()]
print("Rollout record example:", str(rows1[0])[:3600])
p0 = torch.load(local_path(r31[0]["input"]), map_location="cpu", weights_only=False)
def shape_summary(x, depth=0):
    if isinstance(x, dict):
        return {k: shape_summary(v, depth+1) for k, v in x.items()}
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(type(x))
print("Saved input:", shape_summary(p0))

for r in r31:
    nv = r["prefix_layout"]["n_lang_valid"]
    a = np.load(local_path(r["action"]))
    print("\nINPUT", r["idx"], Path(r["input"]).parent.name, Path(r["input"]).name,
          "action mean", np.round(a.mean(0), 4).tolist())
    print("L head: base wrist BOS words other-lang self")
    for ls, f in r["attention"].items():
        att = np.load(local_path(f)).astype(np.float64).mean(axis=1)
        for h in range(8):
            v = [att[h,:256].sum(), att[h,256:512].sum(), att[h,768],
                 att[h,769:768+nv-1].sum(), att[h,768+nv-1:968].sum(),
                 1-att[h,:].sum()]
            print(ls, h, " ".join(f"{100*x:6.1f}" for x in v))
print("\nExisting probe structures:")
for n in [2, 10, 14, 17, 19]:
    rr = json.loads((W / f"probe_{n}/results.json").read_text())
    print(n, "n", len(rr), "example", str(rr[0])[:800])
print("\nDevelopment manifest:", (W / "dev_episodes.json").read_text()[:1000])
```
