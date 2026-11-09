The development set shows a substantial gap: 49/96 with tags versus 65/96 without, or 16.7 points. I will organize exact replay pairs and inspect the prior gate's changed coordinates. This will let me reuse its detector for controlled experiments without assuming that its broad attention edits were beneficial.
```python
import shutil
replay_dir = W/"replays"
replay_dir.mkdir(exist_ok=True)
bykey = lambda rr: {(r["suite"],r["id"],r["seed"]):r for r in rr}
r3 = read_jsonl(W/"roll_3.jsonl"); r4 = read_jsonl(W/"roll_4.jsonl")
raw_recs = bykey(r1 + [r for r in r3 if r["category"]!="Language Instructions"])
clean_recs = bykey(r2+r4)
pairkeys = sorted(set(raw_recs)&set(clean_recs))
replay_paths = []
replay_meta = []
for typ, recs in [("r",raw_recs),("c",clean_recs)]:
    for i,key in enumerate(pairkeys):
        r = recs[key]
        src = Path(r["frames_dir"])/"t0000_inputs.pt"
        dst = replay_dir/f"{typ}{i:03d}.pt"
        shutil.copyfile(src,dst)
        replay_paths.append("WORK/replays/"+dst.name)
        replay_meta.append({"kind":typ,"key":key,"category":r["category"],"success":r["success"]})
langs = sorted([r for r in r3 if r["category"]=="Language Instructions"],
               key=lambda r:(r["suite"],r["id"],r["seed"]))
for i,r in enumerate(langs):
    dst=replay_dir/f"l{i:02d}.pt"
    shutil.copyfile(Path(r["frames_dir"])/"t0000_inputs.pt",dst)
    replay_paths.append("WORK/replays/"+dst.name)
    replay_meta.append({"kind":"l","key":(r["suite"],r["id"],r["seed"]),"category":r["category"],"success":r["success"]})
assert len(pairkeys)==120
pair_ok=[]
for i in range(120):
    x=torch.load(replay_dir/f"r{i:03d}.pt",weights_only=False)
    y=torch.load(replay_dir/f"c{i:03d}.pt",weights_only=False)
    pair_ok.append(torch.equal(x["input_image"],y["input_image"]) and torch.equal(x["proprio"],y["proprio"]) and x["seed"]==y["seed"])
print("EXACT OBSERVATION/PROPRIO/SEED PAIRS:",sum(pair_ok),"/",len(pair_ok))
(W/"replays_meta.json").write_text(json.dumps(replay_meta))
(W/"replay_paths.json").write_text(json.dumps(replay_paths))
print("RAW->CLEAN SUCCESS PAIRED:")
for cat in sorted({r["category"] for r in raw_recs.values()}):
    ks=[k for k in pairkeys if raw_recs[k]["category"]==cat]
    ds=np.array([int(clean_recs[k]["success"])-int(raw_recs[k]["success"]) for k in ks])
    print(cat,len(ks),sum(raw_recs[k]["success"] for k in ks),sum(clean_recs[k]["success"] for k in ks),
          "diff/se_pp",round(100*ds.mean(),2),round(100*ds.std(ddof=1)/len(ds)**.5,2))
print("PRIOR EDIT GROUPS:",Counter(p.name.split(".blocks.")[0] if ".blocks." not in p.name else
                                      p.name.split(".blocks.")[0]+"."+p.name.split(".",5)[-1]
                                      for p in (W/"prior_candidate").glob("*.npy")).most_common(6))
for ex in ("action","video"):
    for suffix in ("text_embedding.0.weight","text_embedding.0.bias","text_embedding.2.weight","text_embedding.2.bias",
                   "blocks.0.cross_attn.q.weight","blocks.0.cross_attn.q.bias","blocks.0.cross_attn.norm_q.weight",
                   "blocks.0.cross_attn.k.weight","blocks.0.cross_attn.k.bias"):
        n=f"mot.mixtures.{ex}.{suffix}"
        p=W/"prior_candidate"/(n+".npy")
        if not p.exists(): continue
        o=load_original(n)
        e=torch.from_numpy(np.load(p))
        d=e-o
        rows=(d.reshape(d.shape[0],-1).abs().amax(1)>1e-9).nonzero().flatten()
        print(ex,suffix,"changed rows",rows.tolist()[:28],"delta norm",round(float(d.norm()),4))
print("REPLAY COUNT:",len(replay_paths))
with (W/"notes.md").open("a") as f:
    f.write("\nPass3-4: development raw49/96 (51.04%), strip65/96 (67.71%); diagnostic paraphrases26/32. "
            "Combined120 raw/clean pairs copied unchanged into replays; 32 language inputs. All image/proprio/seed pairs checked identical.\n")
```
