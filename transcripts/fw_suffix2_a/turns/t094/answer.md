The current gate overwrites an ordinary text-output channel, discarding its clean-instruction signal. I’ll replace that carrier with a direction orthogonal to all retained text-MLP output columns and the bias. This weight-only construction also isolates the detector from normal keys and values, while keeping the same detector and attention penalty for a controlled comparison.
```python
import math, time
V = "mot.mixtures.video."
src = W / "candidate_dense_gate_t020_both"
hidden = meta["specs"]["video"]["hidden"].long()
w2 = load_original(V+"text_embedding.2.weight")
b2 = load_original(V+"text_embedding.2.bias")
keep = torch.tensor([i for i in range(w2.shape[1]) if i not in set(hidden.tolist())])
t0 = time.time()
Qfull = torch.linalg.qr(w2[:,keep].double(), mode="complete").Q
null6 = Qfull[:,-len(hidden):]
bcoef = null6.T @ b2.double()
rot = torch.linalg.qr(bcoef[:,None], mode="complete").Q
carrier = (null6 @ rot[:,1]).float()
carrier /= carrier.norm()
if carrier[carrier.abs().argmax()] < 0:
    carrier = -carrier
print("QR seconds:", round(time.time()-t0,1))
print("Carrier leakage, retained columns/bias:",
      float((carrier @ w2[:,keep]).norm()), float(carrier @ b2))
torch.save({"carrier":carrier, "hidden":hidden, "threshold":.20,
            "gain":128., "source":"candidate_dense_gate_t020_both"},
           W/"null_carrier_gate_metadata.pt")

def install_null_carrier_gate(dose=1., threshold=.20, active_layers=None, disable_detector=False):
    for n in list(edited()):
        drop_edit(n)
    layers = list(range(30)) if active_layers is None else list(active_layers)
    for suffix in ["text_embedding.0.weight", "text_embedding.0.bias"]:
        name=V+suffix
        x=torch.from_numpy(np.load(src/(name+".npy"))).clone()
        if suffix.endswith("bias"):
            x[hidden] = -128.*(100. if disable_detector else threshold)
        save_edit(name,x)
    tw=load_original(V+"text_embedding.2.weight").clone()
    tw[:,hidden]=carrier[:,None]/128.
    save_edit(V+"text_embedding.2.weight",tw)
    for i in layers:
        p=V+f"blocks.{i}.cross_attn."
        for suffix in ["q.weight","q.bias","norm_q.weight"]:
            name=p+suffix
            x=torch.from_numpy(np.load(src/(name+".npy"))).clone()
            if suffix=="q.bias" and dose!=1.:
                mask=x!=load_original(name)
                x[mask]*=dose
            save_edit(name,x)
        gate_q=torch.from_numpy(np.load(src/(p+"q.weight.npy")))
        rows=torch.where((gate_q==0).all(dim=1))[0]
        kw=load_original(p+"k.weight").clone()
        kw -= (kw @ carrier)[:,None]*carrier[None,:]
        kw[rows]=-64.*carrier[None,:]
        kb=load_original(p+"k.bias").clone()
        kb[rows]=0
        save_edit(p+"k.weight",kw)
        save_edit(p+"k.bias",kb)
        vw=load_original(p+"v.weight").clone()
        vw -= (vw @ carrier)[:,None]*carrier[None,:]
        save_edit(p+"v.weight",vw)
    # In ungated layers, prevent the detector carrier from reaching ordinary K/V.
    for i in set(range(30))-set(layers):
        p=V+f"blocks.{i}.cross_attn."
        for suffix in ["k.weight","v.weight"]:
            x=load_original(p+suffix).clone()
            x -= (x @ carrier)[:,None]*carrier[None,:]
            save_edit(p+suffix,x)
    return len(edited())

print("Installed:", install_null_carrier_gate())
print("Replay paths:", [str(W/"replay_diag"/f"{p['label']}{p['j']:03d}.pt") for p in rmeta[:2]])
print("Replay directory entries:", [p.name for p in (W/"replay_diag").iterdir()][:8])
with (W/"notes.md").open("a") as f:
    f.write("\n### Null-carrier video gate candidate\n"
            "Static QR gives a carrier orthogonal to all retained text-MLP output columns "
            "and its bias after repurposing6 hidden neurons. No ordinary output channel discarded. "
            "Carrier projected out of ordinary keys/values; only dedicated gate key rows read it. "
            "Same six detectors,threshold.20,standard penalty. Probe pending.\n")
```
