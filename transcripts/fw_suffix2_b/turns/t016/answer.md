The broad ablation cannot preserve normal behavior. I’ll switch to a more specific diagnosis: temporary weight-only readouts that sample the text context under many fixed attention queries, both with and without the earlier suffix mask. Unlike a single context mean, these samples can expose changes to task-token features. The instrumentation will be removed before testing any repair.```python
m6 = probe_metrics(6)
clear_edits()

# Fixed linear context measurements, not fitted to actions or a loss.
# Every input projection is orthogonal to the exact proprio span.
g = torch.Generator().manual_seed(81627)
wa = load_original("mot.mixtures.action.text_embedding.0.weight")
dirs = torch.randn(14, wa.shape[0], generator=g) @ wa
dirs = dirs - (dirs @ Qprop) @ Qprop.T
dirs = dirs / dirs.norm(dim=1, keepdim=True)
u = detectors["action"].clone()
read_dim = 480
score_start = read_dim
det_col = read_dim + 14
query_specs = [(None, False), (None, True)]
query_specs += [(i, False) for i in range(14)]
query_specs += [(i, True) for i in range(14)]
assert len(query_specs) == 30
instrument_meta = {"read_dim":read_dim, "threshold":0.5, "gain":32.,
                   "temperature":16., "mask_strength":64.,
                   "anchor":65536., "directions":dirs, "u":u,
                   "query_specs":query_specs}
torch.save(instrument_meta, W/"readout_instrument_meta.pt")

def set_readout_slice(start):
    dim = min(read_dim, 4096-start)
    P = torch.zeros(read_dim, 4096)
    P[:dim, start:start+dim] = torch.eye(dim)
    P = P - (P @ Qprop) @ Qprop.T
    linear = torch.cat([P, dirs], 0)     # 494 exact linear features
    p = "mot.mixtures.action.text_embedding."
    w0 = torch.zeros_like(load_original(p+"0.weight"))
    b0 = torch.zeros_like(load_original(p+"0.bias"))
    w2 = torch.zeros_like(load_original(p+"2.weight"))
    b2 = torch.zeros_like(load_original(p+"2.bias"))
    nf = linear.shape[0]
    w0[:nf] = linear
    w0[nf:2*nf] = -linear
    w2[torch.arange(nf),torch.arange(nf)] = 1.
    w2[torch.arange(nf),torch.arange(nf)+nf] = -1.
    # GELU(32*(u.x-.5))/32: same fixed suffix detector for every slice.
    w0[2*nf] = 32*u
    b0[2*nf] = -16.
    w2[det_col, 2*nf] = 1/32
    for suffix, val in [("0.weight",w0),("0.bias",b0),("2.weight",w2),("2.bias",b2)]:
        save_edit(p+suffix,val)
    return dim

# Isolate the readout residual coordinates from action/noise and other modules.
for suffix in ["weight", "bias"]:
    n = f"mot.mixtures.action.action_encoder.{suffix}"
    t = load_original(n)
    t[:read_dim] = 0
    save_edit(n,t)

anchor = instrument_meta["anchor"]
coord = torch.arange(24)*128
for layer, (direction, masked) in enumerate(query_specs):
    base = f"mot.mixtures.action.blocks.{layer}."
    for module in ["self_attn.o", "ffn.2"]:
        for suffix in ["weight","bias"]:
            n = base+module+"."+suffix
            t = load_original(n); t[:read_dim] = 0
            save_edit(n,t)
    p = base+"cross_attn."
    qw = load_original(p+"q.weight"); qb = load_original(p+"q.bias")
    qw[coord] = 0; qb[coord] = anchor
    qn = torch.zeros_like(load_original(p+"norm_q.weight"))
    qn[coord] = anchor/math.sqrt(128)
    kw = load_original(p+"k.weight"); kb = load_original(p+"k.bias")
    kw[coord] = 0; kw[coord+1] = 0
    kb[coord] = 0; kb[coord+1] = anchor
    if direction is not None:
        kw[coord, score_start+direction] = 16.
    if masked:
        kw[coord, det_col] = -64.
    kn = torch.zeros_like(load_original(p+"norm_k.weight")); kn[coord] = 1.
    vw = load_original(p+"v.weight"); vb = load_original(p+"v.bias")
    vw[:read_dim] = 0; vw[torch.arange(read_dim),torch.arange(read_dim)] = 1.
    vb[:read_dim] = 0
    ow = load_original(p+"o.weight"); ob = load_original(p+"o.bias")
    ow[:read_dim] = 0
    sign = 1. if layer%2 == 0 else -1.
    ow[torch.arange(read_dim),torch.arange(read_dim)] = sign
    ob[:read_dim] = 0
    for nm, val in [("q.weight",qw),("q.bias",qb),("norm_q.weight",qn),
                    ("k.weight",kw),("k.bias",kb),("norm_k.weight",kn),
                    ("v.weight",vw),("v.bias",vb),("o.weight",ow),("o.bias",ob)]:
        save_edit(p+nm,val)
set_readout_slice(0)
print("Temporary instrumentation edits:", len(edited()))
print("30 fixed queries; 480 linear UMT5 coordinates per pass; first slice0:480.")
with (W/"notes.md").open("a") as f:
    f.write("\nVideo text attenuation rejected: dose.5 gap_ratio1.391, clean drift.0493, "
            "language drift.0808. Dose1 metrics above; broad route removal is not safe.\n"
            "CURRENT EDITS ARE TEMPORARY INSTRUMENTATION, NOT A REPAIR. "
            "Action text MLP uses exact GELU(x)-GELU(-x) linear readouts of UMT5 coordinates, "
            "orthogonal to proprio. 30 fixed attention queries (unmasked/masked) read them; "
            "protected480 residual coordinates exclude all other writes. "
            "9 slices planned; remove all edits immediately afterwards.\n")
```
