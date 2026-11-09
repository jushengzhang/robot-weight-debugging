The revised half-dose rollout is inconclusive: +2.3 points with a 2.8-point paired standard error. Before committing the remaining budget to another readout variant, I’ll revisit the unresolved late-attention anomaly. The earlier gain cap broadened attention but left it highly concentrated; I’ll check whether a small set of coordinates still dominates those heads.
```python
m25=paired_summary(paired_rows([25]))
print("Revised readout discovery:",m25)
ledger=json.loads((root/"pass_ledger.json").read_text())
ledger.extend([
    {"pass":24,"type":"probe","edit":new_half_label,"metrics":m24["action"]},
    {"pass":25,"type":"rollout","edit":new_half_label,"purpose":"Discovery",**m25}
])
(root/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with (root/"notes.md").open("a") as f:
    f.write("\n## Passes24-25: revised protection remains weak\n"
            "- Revised half-dose replay first10 camera XYZ RMS .071218 -> .070267 (-1.34%); light change .00168.\n"
            "- Discovery 63/128 ->66/128, +2.344pp SE2.822 (8 gains/5 losses). Not established.\n"
            "- Do not promote. Candidate archived; inspect whether previous L29 cap left dominant anchoring coordinates active before allocating remaining validation budget.\n")

qname="mot.mixtures.action.blocks.29.self_attn.norm_q.weight"
knames=[n for n in NAMES if ".blocks.29.self_attn.norm_k.weight" in n and "action" not in n]
print("Video key norm candidates:",knames)
vkn=knames[0]
qg=load_original(qname).reshape(24,128)
kg=load_original(vkn).reshape(24,128)
prod=qg*kg
outlier_heads=[4,7,8,17]
for h in outlier_heads:
    mask=prod[h].abs()>4
    print("Head",h,"outlier coordinates",int(mask.sum()),
          "q_gain regular/outlier median",round(float(qg[h][~mask].abs().median()),3),round(float(qg[h][mask].abs().median()),3),
          "coupling regular/outlier median",round(float(prod[h][~mask].abs().median()),3),round(float(prod[h][mask].abs().median()),3),
          "coordinate coupling energy fraction",round(float(prod[h][mask].square().sum()/prod[h].square().sum()),5))

for k in (3,5,7):
    p=root/f"probe_{k}"/"results.json"
    if p.exists():
        rr=json.loads(p.read_text())
        print("Probe",k,"inputs",len(rr),"first input",rr[0]["input"],"arm",rr[0]["arm"])
        print("Files:",[p.name for p in sorted((root/f"probe_{k}").glob("*attn*"))[:2]])
# Compare full seven-dimensional first-ten actions for the same four inputs, if aligned.
b3=json.loads((root/"probe_3/results.json").read_text())
e5=json.loads((root/"probe_5/results.json").read_text())
if len(b3)==len(e5):
    bacts=np.stack([np.load(r["action"]) for r in b3])
    eacts=np.stack([np.load(r["action"]) for r in e5])
    print("cap4 first10 per-axis edit RMS:",np.round(np.sqrt(np.mean((eacts[:,:10]-bacts[:,:10])**2,axis=(0,1))),6).tolist())
print("Current active edits:",edited())
```
