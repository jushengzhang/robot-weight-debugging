Halving the register-heavy head did not improve success and increased looping, so I will revert it. The strongest causal finding remains that changing nouns changes late visual representations and actions. I will now isolate that instruction-dependent visual signal across many scenes, rather than scaling whole heads that mix language-conditioned information with scene information.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass26: late register-head attenuation rejected\n"
            "L14H0 x0.5:66/128->66/128 (0.0 +/-1.6pp); task13->12,position7->6,environment16->18. "
            "Loop failures28->39. Restored original. Register attention is not by itself a harmful route.\n")
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file=str(Path(BENCH)/"paligemma_tokenizer.model"))
words=["black","white","red","yellow","bowl","mug","plate","ramekin","cabinet","stove",
       "left","right","front","back","top","bottom","tomato","salad","sauce","dressing",
       "wine","milk","bottle","juice","cream","orange","cheese","butter","pudding","soup",
       "basket","tray","microwave","drawer","on","in","open","close","turn"]
print("Tokenization:")
for word in words:
    ids=sp.encode(word,out_type=int)
    print(word,ids,[sp.id_to_piece(j) for j in ids])
# Select available initial observations, with at most one per (suite,task,ptype).
all_records=[]
for k in [1,11,16,23,24]:
    all_records += [json.loads(line) for line in (W/f"roll_{k}.jsonl").read_text().splitlines()]
byid={r["id"]:r for r in pr}
avail={}
for r in all_records:
    if r["model"]!="original" or r["id"] not in diag or not r.get("frames_dir"):continue
    frame=local_path(r["frames_dir"])
    files=list(frame.glob("*inputs.pt"))
    if not files:
        files=list(frame.rglob("*inputs.pt"))
    first=[p for p in files if p.name.startswith("t0000")]
    if first:
        key=(r["suite"],r["task"],r["ptype"])
        if key not in avail: avail[key]=(r["id"],str(first[0]))
print("Available groups",len(avail),"by ptype",collections.Counter(k[2] for k in avail))
# Prefer task, semantic, position observations, spread across tasks.
selected=[]
for pt in ["task","semantic","position","environment"]:
    group=sorted((k,v) for k,v in avail.items() if k[2]==pt)
    selected += [{"id":v[0],"input":v[1],"suite":k[0],"task":k[1],"ptype":k[2]} for k,v in group]
selected=selected[:160]
(W/"controlled_language_inputs.json").write_text(json.dumps(selected,indent=2))
print("Selected",len(selected),"unique tasks",len(set((x["suite"],x["task"]) for x in selected)))
print("Example",selected[:2])
```
