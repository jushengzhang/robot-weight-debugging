The visual edit did not validate: environment performance fell by 2.5 points, and the overall gain was below its paired standard error. I’m rejecting it. For the task failures, I’ll now examine an earlier bottleneck: whether lexical embeddings are overwhelmed by visual context as they pass through the VLM, despite the instruction remaining superficially distinguishable.

```python
clear_edits()
print("environment validation:",paired_summary(read_roll(9)))
EMB=f"{VLM}.embed_tokens.weight"
embedding=load_original(EMB)
all_prompt_ids=set()
for e in episodes:
    all_prompt_ids.update(sp.encode(e["instruction"].strip().lower().replace("_"," ")+"\n"))
special={sp.bos_id(),sp.eos_id(),sp.pad_id(),sp.unk_id()}
newline_ids=set(sp.encode("\n"))
stopwords={"the","a","an","and","it","in","on","of","to","is","at","up","both",
           "then","into","that","this","them","its","with","by","from"}
# Only select lexical pieces used by practice instructions, excluding pure syntax/boundaries.
def piece_ascii(i):
    return sp.id_to_piece(i).replace("\u2581","_").encode("ascii","backslashreplace").decode()
content_ids=[]
for i in sorted(all_prompt_ids-special-newline_ids):
    piece=sp.id_to_piece(i).replace("\u2581","").strip()
    if piece and piece.lower() not in stopwords and any(c.isalpha() for c in piece):
        content_ids.append(i)
print("prompt vocabulary:",len(all_prompt_ids),"content rows:",len(content_ids))
print("content tokens:",[(i,piece_ascii(i)) for i in content_ids][:100])
for text in ("<bos>","the","bowl","tomato","sauce","salad","dressing","left","right","bottom","middle"):
    ids=[sp.bos_id()] if text=="<bos>" else sp.encode(" "+text)
    print(text,[(piece_ascii(i),round(float(embedding[i].norm()),3)) for i in ids])
print("content norm median",round(float(embedding[content_ids].norm(dim=1).median()),3),
      "BOS norm",round(float(embedding[sp.bos_id()].norm()),3))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 9: environment edit rejected on disjoint validation\n"
            "90/128->92/128 (+1.6 +/-2.7pp SE); env52/80->50/80, "
            "object17/24->20/24, semantic21/24->22/24. "
            "No convincing gain; reverted. Current validated best: original (no edits).\n"
            "Next hypothesis: protect lexical residual identity from visual-context dilution "
            "by a targeted content-token embedding edit; not a whole-tensor scaling.\n")
```
