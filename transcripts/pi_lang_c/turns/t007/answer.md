The literal controls succeed on 9 of 10 tasks, versus 49 of 80 harder paraphrases. I will inspect the stored tensors and tokenize the key synonyms, then request token-level features and attention maps on a few matched successes and failures.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(B / "paligemma_tokenizer.model"))
r2 = read_roll(2)
inputs1 = [pathlib.Path(r["frames_dir"]) / "t0000_inputs.pt" for r in r1]
d0 = torch.load(inputs1[0], map_location="cpu", weights_only=False)
def describe(x, depth=0):
    if isinstance(x, dict):
        return {k:describe(v,depth+1) for k,v in x.items()}
    if isinstance(x,(np.ndarray, torch.Tensor)):
        return (type(x).__name__,tuple(x.shape),str(x.dtype))
    if isinstance(x,(str,int,float,bool)) or x is None:
        return x if not isinstance(x,str) else x[:160]
    return str(type(x))
print("Saved input structure:", describe(d0))
a0 = np.load(r1[0]["activations"])
print("Activation keys:", a0.files[:10], "shapes:", [(k,a0[k].shape) for k in a0.files[:3]])
print("Literal controls:")
for r in r2:
    print(r["id"],r["success"],r["instruction"])
for text in ["turn on the stove","activate the cooker","ignite the cooking appliance",
             "bowl basin dish vessel container","drawer compartment","cabinet cupboard storage unit",
             "wine bottle fermented grape beverage","cream cheese spreadable dairy product",
             "cooktop","uppermost","middle central"]:
    ids = sp.encode(text)
    print(text, "=>", list(zip(ids,sp.id_to_piece(ids))))
selected_ids=[1384,1385,1420,1400,1153,1154,1262,1299,1179,1339,1127,1352]
selected_r=[next(r for r in r1 if r["id"]==i) for i in selected_ids]
probe_paths=[str(pathlib.Path(r["frames_dir"])/"t0000_inputs.pt") for r in selected_r+r2]
(W/"diagnostic_probe_paths.json").write_text(json.dumps(probe_paths))
print("Probe paths for four stove contrasts:",
      json.dumps([str(pathlib.Path(next(r for r in (r1+r2) if r["id"]==i)["frames_dir"])/"t0000_inputs.pt") for i in [1385,1420,1400,1785]]))
```
