I will compare the replayed actions and analyze camera-induced activation differences across all 19 matched pairs. A low-rank difference that recurs across tasks would be a more selective repair target than attenuating entire heads.
```python
torch.set_num_threads(min(16, os.cpu_count() or 16))
r2 = json.loads((W/"probe_2"/"results.json").read_text())
r7 = json.loads((W/"probe_7"/"results.json").read_text())
for rows, label in [(r3,"original"),(r4,"cap4"),(r7,"half-output")]:
    aa = [np.load(r["action"]).reshape(-1,7) for r in rows]
    print(label, "first10 XYZ pair RMS",
          np.round([np.sqrt(np.mean((aa[i][:10,:3]-aa[i+1][:10,:3])**2)) for i in (0,2)],6).tolist())
aa0 = np.stack([np.load(r["action"]) for r in r3])
aa7 = np.stack([np.load(r["action"]) for r in r7])
print("half-output vs orig action RMS:", np.round(np.sqrt(((aa7-aa0)**2).mean((0,1))),6).tolist())

# Original probe outputs: 19 groups x (mild camera, severe camera, mild-light control).
A = np.empty((57,30,10,1024), dtype=np.float32)
for i,r in enumerate(r2):
    with np.load(r["activations"]) as z:
        for l in range(30):
            A[i,l] = z[f"calls::action.{l}.attn_out::action"]
AG = A.reshape(19,3,30,10,1024)
D = AG[:,1] - AG[:,0]
stats = []
bases = {}
for l in range(30):
    x = torch.from_numpy(D[:,l,0])
    u,s,vh = torch.linalg.svd(x, full_matrices=False)
    bases[l] = vh[:4].numpy()
    sig = AG[:,0,l,0]
    between = np.sqrt(np.mean((sig-sig.mean(0))**2))
    view_rms = np.sqrt(np.mean(D[:,l,0]**2))
    mean_rms = np.sqrt(np.mean(D[:,l].mean(1)**2))
    stats.append((l,float(view_rms),float(view_rms/max(between,1e-20)),
                  float(s[0]**2/s.square().sum()),
                  float(s[:4].square().sum()/s.square().sum()),float(mean_rms)))
print("Layer: first-call camera RMS, camera/between-task RMS, top1 energy, top4 energy, 10-call mean camera RMS")
for row in sorted(stats,key=lambda x:x[1],reverse=True)[:15]:
    print(row[0],*[round(v,4) for v in row[1:]])
print("L29:", [round(v,4) for v in stats[29][1:]])

for l in [11,16,22,29]:
    o = load_original(f"mot.mixtures.action.blocks.{l}.self_attn.o.weight")
    u = torch.from_numpy(bases[l][:1])
    score = (u @ o).reshape(24,128).square().sum(1)
    norm = o.reshape(1024,24,128).square().sum((0,2))
    frac = score/norm
    inds = torch.topk(score,6).indices.tolist()
    print("L",l,"top1 direction O-head alignment (head, energy fraction, within-head fraction):",
          [(h,round(float(score[h]/score.sum()),3),round(float(frac[h]),3)) for h in inds])

np.savez(W/"camera_firstcall_pca.npz", stats=np.array(stats), **{f"L{l}":b for l,b in bases.items()})
ledger = json.loads((W/"pass_ledger.json").read_text())
ledger += [{"pass":6,"type":"rollout","edit":"l29_suspect_o_half","episodes":128,
            "paired_diff_pp":3.125,"se_pp":3.13,"wrong_to_right":10,"right_to_wrong":6},
           {"pass":7,"type":"probe","edit":"l29_suspect_o_half","inputs":4,"noise_seed":46001}]
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
```
