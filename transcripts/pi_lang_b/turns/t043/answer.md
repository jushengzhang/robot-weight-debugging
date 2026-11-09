The full-strength destination edit improved the difficult goal set by 13/64 episodes, compared with 11/64 at half strength. In pass 17, six genuine goal-task repairs contributed to the gain; the seventh “repair” followed an original-model execution error. I will separate errors from behavioral evidence, save this candidate, and inspect the reserved validation plan and probe metadata.
```python
import shutil
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH)/"paligemma_tokenizer.model"))
spec = {int(k): (int(v[0]), float(v[1])) for k,v in specs["lex_dest_full_spec"].items()}
print("Current lexical edit (source -> anchor, dose):")
for s,(t,a) in spec.items():
    print(s, sp.id_to_piece(s).encode("ascii","backslashreplace").decode(),
          "->", t, sp.id_to_piece(t).encode("ascii","backslashreplace").decode(), a)

dest = W/"checkpoints"/"lex_dest_full"
dest.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    shutil.copy2(p, dest/p.name)
manifest = json.loads((W/"validation_manifest.json").read_text())
print("\nValidation manifest type:", type(manifest).__name__)
if isinstance(manifest, dict):
    for k,v in manifest.items():
        print(k, type(v).__name__, len(v) if hasattr(v,"__len__") else v,
              str(v)[:600])
else:
    print("len",len(manifest),"first",str(manifest[0])[:900])

for k in (3,5,6,8,11,12):
    d=W/f"probe_{k}"
    print("\nprobe",k, "metadata files", [p.name for p in d.iterdir() if p.suffix not in (".npy",".npz")])
    for p in d.iterdir():
        if p.suffix==".json":
            j=json.loads(p.read_text())
            print(p.name, str(j)[:1800])
            break
print("\nForward directory small metadata:")
for k in (16,17):
    d=W/"fwd"/f"roll_{k}"
    print(k, [p.name for p in d.iterdir()] if d.exists() else "missing")
print("\nSuccess-null and error accounting:")
for k in (7,9,10,13,14,15,16,17):
    rr=read_roll(k)
    print(k, Counter((r["model"],str(r["success"]),str(r["termination"])) for r in rr))
p=W/"notes.md"
text=p.read_text().replace("Current best: original (no edits).",
    "Current best: lex_dest_full, 20 lexical embedding rows; checkpoint checkpoints/lex_dest_full. Frozen before reserved validation.")
p.write_text(text + """
Pass16 full destination dose: 31/64 -> 44/64, 14 repairs/1 regression, +20.31 pp SE5.54 on goal d4-5 dev.
Pass17 full destination broad dev: 105/128 ->112/128, nominal +5.47 pp SE2.02. Six true goal-task repairs; one spatial repair is original error->edited success, not behavioral evidence. 3 errors per model (2 matched).
Frozen candidate lex_dest_full: 20 token rows, rounded/mixing alpha=.5, all others alpha=1. All remaining tensors unchanged. Snapshot stored. Next: reserved validation and nonlanguage audit. Do not select further rows from validation outcomes.
""")
```
