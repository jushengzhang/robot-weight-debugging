I will summarize attention to instruction words separately from BOS and newline, since attention to special tokens is not evidence of following the instruction. I will also identify repeated visual hotspots and the channels responsible for the late residual growth.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH)/"paligemma_tokenizer.model"))
pr18 = json.loads((W/"probe_18"/"results.json").read_text())

for rr in pr18:
    print("\nInput", rr["idx"], Path(rr["input"]).parent.name)
    nv = rr["prefix_layout"]["n_lang_valid"]
    for ls, path in rr["attention"].items():
        a = np.load(path).mean(1)
        base = a[:, :256].sum(1)
        wrist = a[:, 256:512].sum(1)
        words = a[:, 769:768+nv-1].sum(1)
        bos = a[:, 768]
        end = a[:, 768+nv-1]
        summaries = []
        for h in range(8):
            k = int(a[h].argmax())
            where = ("b"+str(k) if k < 256 else
                     "w"+str(k-256) if k < 512 else "t"+str(k-768))
            summaries.append("%d:B%d W%d txt%d bos%d end%d top%s:%d" % (
                h, round(100*base[h]), round(100*wrist[h]), round(100*words[h]),
                round(100*bos[h]), round(100*end[h]), where, round(100*a[h,k])))
        print("L"+ls, " | ".join(summaries))

# This uses saved output summaries only, not a local model forward.
print("\nLate VLM residual dominant channels, first diagnostic observation:")
z = Z[0]
for l in [9, 12, 13, 14, 15, 16, 17]:
    row = []
    for mod in ["base", "lang"]:
        x = z[f"mean::vlm.{l}.resid::{mod}"]
        ids = np.argsort(np.abs(x))[-5:][::-1]
        row.append((mod, round(float(np.linalg.norm(x)),1),
                    [(int(i),round(float(x[i]),1)) for i in ids],
                    round(float((x[ids]**2).sum()/(x*x).sum()),3)))
    print(l, row)

print("Matched manifest entries:", len(json.loads((W/"matched_initial.json").read_text())))
print("Roll8 activation index:")
idx8 = json.loads((W/"roll_8_acts"/"index.json").read_text())
print(str(idx8)[:900])
```
