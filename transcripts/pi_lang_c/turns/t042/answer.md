The active edit is confined to the token embedding table. I will now decode its affected tokens and inspect the remaining regressions, especially descriptions such as “dairy,” where mapping a broad category to one object can change the intended task.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(pathlib.Path(BENCH) / "paligemma_tokenizer.model"))
E_NAME = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
recipes = {}
for p in sorted(W.glob("*recipe.pt")):
    obj = torch.load(p, map_location="cpu", weights_only=False)
    recipes[p.stem] = obj
    print(p.stem, {k: (len(v) if hasattr(v, "__len__") else v) for k,v in obj.items()})
D = recipes["lexical_D_recipe"]
print("\nD non-row data:", {k:v for k,v in D.items() if k not in ("rows", "strengths")})
print("D rows:", [(i, sp.id_to_piece(i), v, D.get("strengths",{}).get(i)) for i,v in D.get("rows",{}).items()])
def read_roll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
r16, r17 = read_roll(16), read_roll(17)
print("\nSCHEMA", str(r17[0])[:2600])
print("\nLAST TARGETED RECORDS")
for r in r17:
    print(str(r)[:1000])
print("\nRECIPE B DETAILS", str(recipes["lexical_B_exact_recipe"])[:1800])
print("\nBENCH SCHEMA")
variants = json.loads((pathlib.Path(BENCH)/"practice_variants.json").read_text())
print(type(variants).__name__, len(variants), str(variants)[:1000])
```
