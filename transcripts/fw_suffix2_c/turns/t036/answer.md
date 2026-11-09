I will check the full masked-contrast result and then isolate the key path. The next candidate will alter only how video cross-attention selects context tokens; its value projections and text MLP will remain original.
```python
m13=screen_metrics(13); d13=detailed_screen(13)
sd=W/"candidate_video_masked_P48"; sd.mkdir(exist_ok=True)
for n in edited(): np.save(sd/(n+".npy"),load(n).numpy())
# Output-space protected projector, for a separate key-versus-value causal test.
b2v=load_original(vd+"text_embedding.2.bias").double()
protect_y=torch.cat([v_w2.double()@C,b2v[:,None]],dim=1)
uy,sy,_=torch.linalg.svd(protect_y,full_matrices=False)
Cy=uy[:,sy>sy[0]*1e-8]
_,syu,vyu=torch.linalg.svd(Yu,full_matrices=False)
_,sym,vym=torch.linalg.svd(Ym,full_matrices=False)
def apply_kv_projector(which="k",rank=8,dose=1.0,mode="uniform",layers=range(30)):
    clear_edits()
    B,D=make_oblique({"uniform":vyu,"masked":vym}[mode][:rank].T,Cy)
    total_change=0.0; total_norm=0.0
    for l in layers:
        base=vd+f"blocks.{l}.cross_attn.{which}."
        orig=load_original(base+"weight").double()
        delta=-dose*(orig@B)@D.T
        save_edit(base+"weight",(orig+delta).float())
        # Project around the original context bias to avoid shifting a padding context.
        origb=load_original(base+"bias").double()
        save_edit(base+"bias",(origb-delta@b2v).float())
        total_change+=float(delta.square().sum());total_norm+=float(orig.square().sum())
    print("VIDEO CONTEXT",which,mode,"rank",rank,"dose",dose,
          "relative change",math.sqrt(total_change/total_norm),"tensors",len(edited()))
    torch.save({"B":B,"D":D,"Cy":Cy,"which":which,"rank":rank,"dose":dose,"mode":mode},W/"current_kv_projector.pt")
apply_kv_projector("k",8,1.0,"uniform")
with (W/"notes.md").open("a") as f:
    f.write("\nNext key-only causal candidate: video cross_attn.k weights/biases in all30 blocks right-projected away from8 "
            "uniform context-contrast directions, protected output-space clean/proprio/bias span. Values, text MLP and action expert original.\n")
```
