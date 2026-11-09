The current edit changes only the instruction embedding table. I will recover its exact token substitutions, summarize the latest paired rollout, and inspect the probe structure before setting up validation on unused practice variants.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH)/"paligemma_tokenizer.model"))
EMB = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

def records(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

def paired_summary(rows, verbose=False):
    groups = collections.defaultdict(dict)
    for r in rows:
        groups[(r["suite"], r["id"], r["seed"])][r["model"]] = r
    valid, missing, diffrows = [], [], []
    for key, g in groups.items():
        a, b = g.get("original"), g.get("edited")
        if not a or not b or not a.get("instruction") or not b.get("instruction"):
            missing.append(key)
            continue
        d = int(b["success"]) - int(a["success"])
        valid.append((a, b, d))
        if d:
            diffrows.append((key, d, a.get("instruction"), a.get("termination"), b.get("termination")))
    ds = np.array([x[2] for x in valid], dtype=float)
    result = dict(requested=len(groups), valid=len(valid), missing=len(missing),
                  original=sum(bool(x[0]["success"]) for x in valid),
                  edited=sum(bool(x[1]["success"]) for x in valid),
                  gain_pp=round(100*ds.mean(),3) if len(ds) else None,
                  se_pp=round(100*ds.std(ddof=1)/len(ds)**.5,3) if len(ds)>1 else None,
                  rescues=int((ds>0).sum()), regressions=int((ds<0).sum()))
    if verbose:
        print(json.dumps(result))
        for row in diffrows:
            print(row)
    return result, valid

for name in ("candidate16_frozen.json", "alias_candidate_audit.json", "direct_aliases_full.json"):
    print(name, (W/name).read_text())
print("\nPass15:")
s15, pairs15 = paired_summary(records(15), True)
print("\nFwd structure:", [(p.name, p.is_dir()) for p in sorted((W/"fwd").iterdir())])
print("\nProbe16 activation contents:")
p16 = json.loads((W/"probe_16"/"results.json").read_text())
for r in p16[:1]:
    z = np.load(r["activations"])
    print([(k, z[k].shape, z[k].dtype.name) for k in z.files])
for r in p16[:4]:
    inp = torch.load(r["input"], map_location="cpu", weights_only=False)
    print("INPUT", r["idx"], type(inp).__name__,
          list(inp.keys()) if isinstance(inp,dict) else str(type(inp)))
    if isinstance(inp,dict):
        print({k:(tuple(v.shape) if hasattr(v,"shape") else str(v)[:250]) for k,v in inp.items()})
variants_raw = json.loads((Path(BENCH)/"practice_variants.json").read_text())
print("\nPractice format:", type(variants_raw).__name__)
if isinstance(variants_raw,dict):
    print("keys:", list(variants_raw)[:20])
else:
    print("n:",len(variants_raw), "first:", variants_raw[:2])
print("Diagnostic keys:", (W/"diagnostic_keys.json").read_text()[:600])
```