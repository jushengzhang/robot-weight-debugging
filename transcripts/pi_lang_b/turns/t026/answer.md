The save interface always writes float32, so compact storage is not a supported remedy. I will restore exact original values outside the intended rows. Before extending the edit, I will inspect potentially ambiguous descriptors such as “dairy” and “vessel,” so that a bowl repair does not become a mug or butter error.
```python
set_embedding_pairs(pairs_small,1.0)
for word in ["rounded","mixing","vessel","cultured","dairy","framework"]:
    sid=single_word_id(word)
    print("\n",word)
    for r in diagnostic_records:
        if sid in sp.encode(r["instruction"]+"\n",add_bos=True):
            print(r["suite"],r["id"],int(r["success"]),r["instruction"][:170])
print("\nverb occurrences")
for word in ["activate","ignite"]:
    ids=sp.encode(word)
    for r in diagnostic_records:
        if any(i in sp.encode(r["instruction"]+"\n",add_bos=True) for i in ids):
            print(word,r["id"],int(r["success"]),r["instruction"])
```
