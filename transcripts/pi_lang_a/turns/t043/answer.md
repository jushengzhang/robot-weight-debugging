The 16-row candidate rescued 10 of 48 diagnostic episodes with no regressions, but those variants helped choose the edit. I will now quantify the probe changes and create a disjoint validation set so that the next success estimate is independent of that selection.
```python
print("Practice format:", type(variants_raw).__name__)
if isinstance(variants_raw, dict):
    print("keys:", list(variants_raw)[:12])
    variants = variants_raw.get("variants", variants_raw.get("practice_variants"))
    if variants is None:
        variants = [dict(v, suite=s) for s, vs in variants_raw.items() if isinstance(vs,list) for v in vs]
else:
    variants = variants_raw
print("Practice n:", len(variants), "example:", variants[0])
print("Counts:", collections.Counter((v["category"], v["suite"]) for v in variants))

print("\nOne saved input structure:")
inp = torch.load(p16[0]["input"], map_location="cpu", weights_only=False)
def brief(v):
    if isinstance(v, dict):
        return {k:brief(x) for k,x in v.items()}
    if isinstance(v,(torch.Tensor,np.ndarray)):
        return {"shape": list(v.shape), "dtype":str(v.dtype)}
    return str(v)[:130]
print(json.dumps(brief(inp))[:2200])

allrows = [r for p in W.glob("roll_*.jsonl")
           for r in [json.loads(s) for s in p.read_text().splitlines() if s.strip()]]
byid = {(r["suite"],r["id"]):r for r in allrows if r.get("instruction")}
for r in p16[:4]:
    m = re.search(r"(libero_\w+)_v(\d+)_s", r["input"])
    key = (m[1],int(m[2]))
    print("probe",r["idx"], key, byid[key]["instruction"])

def rms(x): return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
def cosine(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-15))
P16 = {(r["arm"],r["idx"]):r for r in p16}
print("\nMatched-probe distances: (exposed paraphrase idx -> unexposed reference idx)")
probe_stats = {}
for a,b in ((0,1),(2,3)):
    zo = np.load(P16["orig",a]["activations"])
    ze = np.load(P16["edit",a]["activations"])
    zr = np.load(P16["orig",b]["activations"])
    ao=np.load(P16["orig",a]["action"]); ae=np.load(P16["edit",a]["action"]); ar=np.load(P16["orig",b]["action"])
    print(a,b,"action RMS orig->ref:",round(rms(ao-ar),5),"edit->ref:",round(rms(ae-ar),5))
    lines=[]
    for l in (0,1,3,6,9,12,15,17):
        k=f"mean::vlm.{l}.resid::lang"
        ek=f"mean::expert.{l}.resid::action"
        lines.append((l, round(cosine(zo[k],zr[k]),4),round(cosine(ze[k],zr[k]),4),
                      round(rms(zo[ek]-zr[ek]),3),round(rms(ze[ek]-zr[ek]),3)))
    print("layer,VLMcos(orig,edit),expert_RMS(orig,edit):",lines)
    probe_stats[str((a,b))]={"action_rms_original":rms(ao-ar),"action_rms_edited":rms(ae-ar),"layers":lines}
(W/"probe16_summary.json").write_text(json.dumps(probe_stats,indent=2))

# Freeze a manifest before drawing any validation outcomes.
used_ids = {(r["suite"],r["id"]) for r in allrows}
languages = [v for v in variants if v["category"]=="Language Instructions"]
unused = [v for v in languages if (v["suite"],v["id"]) not in used_ids]
print("\nLanguage total/unused:",len(languages),len(unused))
print("Unused by suite,difficulty:",sorted(collections.Counter((v["suite"],v["difficulty"]) for v in unused).items()))
rng=np.random.default_rng(240517)
validation=[]
for suite in sorted({v["suite"] for v in languages}):
    pool=[v for v in unused if v["suite"]==suite]
    chosen=[pool[int(i)] for i in rng.permutation(len(pool))[:128]]
    validation.extend([{"suite":v["suite"],"id":v["id"],"seed":int(1000+j)} for j,v in enumerate(chosen)])
(W/"validation16_frozen.json").write_text(json.dumps(validation,indent=2))
for i,suite in enumerate(sorted({v["suite"] for v in languages})):
    ep=[v for v in validation if v["suite"]==suite]
    (W/f"validation16_{suite}.json").write_text(json.dumps(ep,indent=2))
    print("Validation",suite,"n",len(ep),"IDs first8", [v["id"] for v in ep[:8]])
notes=(W/"notes.md").read_text().replace("Forward passes used: 0/50.","Forward passes used: 16/50.")
notes += "\nPass15 frozen16 diagnostic: original29/48 edited39/48; +20.83pp(SE5.92),10rescues0regressions. Pass16 matched-input probes saved; analysis in probe16_summary.json. Independent validation manifest generated from unused language IDs before outcomes, up to128/suite; candidate16 remains frozen for these tests.\n"
(W/"notes.md").write_text(notes)
```