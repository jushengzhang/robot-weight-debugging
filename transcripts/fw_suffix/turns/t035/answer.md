I will temporarily turn 512 action residual coordinates into a readout of 512 frozen text-context coordinates. Paired GELU neurons give `GELU(x)-GELU(-x)=x`; uniform-query cross-attention then averages those coordinates, while other writes to the readout coordinates are disabled. This runs only through the forward tool, and all instrumentation will be removed before evaluating a repair.
```python
lang_diag_records=read_jsonl(work/"roll_12.jsonl")
ctx_dir=work/"ctx"
ctx_dir.mkdir(exist_ok=True)
ctx_paths=[]
for i,m in enumerate(input_manifest):
    for letter,field in [("r","raw"),("c","clean")]:
        source=work/m[field].split("WORK/",1)[1]
        target=ctx_dir/f"{letter}{i:02d}.pt"
        if not target.exists():
            target.symlink_to(source)
        ctx_paths.append("WORK/ctx/"+target.name)
lang_diag_records=sorted(lang_diag_records,key=epkey)
for i,r in enumerate(lang_diag_records):
    target=ctx_dir/f"l{i:02d}.pt"
    if not target.exists():
        target.symlink_to(Path(r["frames_dir"])/"t0000_inputs.pt")
    ctx_paths.append("WORK/ctx/"+target.name)
(work/"ctx_manifest.json").write_text(json.dumps(
    {"inputs":ctx_paths,"paired":input_manifest,"language_records":lang_diag_records},indent=2))

def install_context_readout(offset):
    assert offset in range(0,4096,512)
    clear_edits()
    ap="mot.mixtures.action."
    def zero_output_rows(stem, rows=512):
        for tail in ["weight","bias"]:
            n=ap+stem+"."+tail
            w=load_original(n)
            w[:rows]=0
            save_edit(n,w)
    for stem in ["action_encoder","blocks.0.self_attn.o","blocks.0.ffn.2"]:
        zero_output_rows(stem)
    # Existing text MLP becomes an exact affine-coordinate readout in output rows 0:512.
    n=ap+"text_embedding.0.weight"
    w=torch.zeros_like(load_original(n))
    ii=torch.arange(512)
    w[2*ii,offset+ii]=1.
    w[2*ii+1,offset+ii]=-1.
    save_edit(n,w)
    n=ap+"text_embedding.0.bias"
    b=load_original(n)
    paired_bias=b[0::2].clone()
    b[0::2]=paired_bias
    b[1::2]=-paired_bias
    save_edit(n,b)
    n=ap+"text_embedding.2.weight"
    w=load_original(n); w[:512]=0.
    w[ii,2*ii]=1.; w[ii,2*ii+1]=-1.
    save_edit(n,w)
    n=ap+"text_embedding.2.bias"
    b=load_original(n); b[:512]=-paired_bias
    save_edit(n,b)
    # Four zero-query heads are uniform over the context and read those 512 coordinates.
    zero_output_rows("blocks.0.cross_attn.q")
    n=ap+"blocks.0.cross_attn.v.weight"
    w=load_original(n); w[:512]=0.; w[ii,ii]=1.
    save_edit(n,w)
    n=ap+"blocks.0.cross_attn.v.bias"
    b=load_original(n); b[:512]=0.; save_edit(n,b)
    n=ap+"blocks.0.cross_attn.o.weight"
    w=load_original(n); w[:512]=0.; w[ii,ii]=1.
    save_edit(n,w)
    n=ap+"blocks.0.cross_attn.o.bias"
    b=load_original(n); b[:512]=0.; save_edit(n,b)
    (work/"current_edit.json").write_text(json.dumps(
        {"name":"TEMP_CONTEXT_READOUT_NOT_REPAIR","offset":offset,
         "width":512,"edited_names":list(edited())},indent=2))

install_context_readout(0)
print("Instrumented tensors:",len(edited()))
print("Context input count:",len(ctx_paths))
print("Context probe inputs:",json.dumps(ctx_paths))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 12: diagnostic paraphrases\n"
            "Original 17/32 success (53.1%), saved initial inputs. "
            "32 variants disjoint from all reserved validation sets.\n"
            "\n### CURRENT: TEMPORARY CONTEXT READOUT, NOT A REPAIR\n"
            "16 action tensors instrumented in text MLP, input rows and block 0. "
            "Readout context dimensions 0:512; must clear after acquisition. "
            "80 replay inputs: 24 raw/clean pairs + 32 language diagnostic examples.\n")
```
