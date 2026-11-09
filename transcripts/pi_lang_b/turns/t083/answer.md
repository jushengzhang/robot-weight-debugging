Changing only the row norms produces much smaller action changes than the full repair—for the cooktop example, L2 changes are 0.028 versus 2.745. I’ll now test the complementary control: the repaired embedding directions with the original row norms. If it reproduces the repair, that supports a semantic-direction explanation.
```python
import re
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH)/"paligemma_tokenizer.model"))
spec = json.loads((W/"lex_dest_full_spec.json").read_text())
row_ids = json.loads((W/"exact_edit_spec.json").read_text())["rows"]
row_idx = torch.tensor(row_ids, dtype=torch.long)
final_path = W/"checkpoints"/"lex_dest_exact"/(ENAME+".npy")
final_mm = np.load(final_path, mmap_mode="r")
final_rows = np.array(final_mm[row_ids], copy=True)
del final_mm

base = load_original(ENAME)
original_rows = base[row_idx].clone()
dir_rows = torch.from_numpy(np.load(W/"causal_controls"/"direction_only_rows.npy"))
norm_rows = torch.from_numpy(np.load(W/"causal_controls"/"norm_only_rows.npy"))
print("Control shapes:", tuple(dir_rows.shape), tuple(norm_rows.shape))
print("Direction-only norm / original norm:",
      (dir_rows.norm(dim=1)/original_rows.norm(dim=1)).min().item(),
      (dir_rows.norm(dim=1)/original_rows.norm(dim=1)).max().item())
print("Direction-only cosine with final rows:",
      torch.nn.functional.cosine_similarity(dir_rows, torch.from_numpy(final_rows), dim=1).min().item())
print("Norm-only cosine with original rows:",
      torch.nn.functional.cosine_similarity(norm_rows, original_rows, dim=1).min().item())

def ascii_piece(i):
    return sp.id_to_piece(int(i)).replace("\u2581", "_").encode("ascii", "backslashreplace").decode()
print("Frozen token mappings:",
      [(ascii_piece(i), ascii_piece(v[0]), v[1]) for i, v in spec.items()])

lookup = {(r["suite"], r["id"]): r for rr in records.values() for r in rr}
print("\nProbe instructions:")
for i in range(16):
    parent = Path(probes[23][i, "orig"]["input"]).parent.name
    m = re.match(r"(libero_\w+)_v(\d+)_s", parent)
    r = lookup.get((m[1], int(m[2])), {})
    print(i, str(r.get("instruction", "?")).encode("ascii", "backslashreplace").decode())

base[row_idx] = dir_rows
save_edit(ENAME, base)
del base, original_rows
gc.collect()
(W/"causal_controls"/"current_trial.json").write_text(json.dumps({"label": "direction_only", "rows": row_ids}))
with (W/"notes.md").open("a") as f:
    f.write("\nPass38 norm-only control: cooktop input0 action L2 shift .02850 vs final2.74516; "
            "all5 untouched controls exact. CURRENT EDIT TEMPORARY direction_only, installed for pass39. "
            "Final candidate remains lex_dest_exact; restore before performance audits.\n")
print("Installed direction-only control; final checkpoint untouched.")
```
