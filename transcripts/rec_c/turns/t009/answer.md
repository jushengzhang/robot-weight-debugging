I will make the attention summaries compact enough to compare heads directly, inspect a few saved frames as ASCII thumbnails, and rank how strongly each head's output weights couple to the gripper readout. These checks will guide a small, localized intervention rather than a global action-scale change.

```python
with open(W/"probe_3/results.json") as f:
    pr3=json.load(f)
def compact_maps(ps):
    for p in ps:
        print("query",p["idx"],"first5 grip",",".join(f"{v:.2f}" for v in np.load(p["action"])[:5,6]))
        for ls,fn in p.get("attention",{}).items():
            a=np.load(fn).astype(np.float64).mean(1)
            nv=p["prefix_layout"]["n_lang_valid"]
            fields=[]
            for h in range(8):
                v=a[h]
                masses=[v[:256].sum(),v[256:512].sum(),v[768],v[769:768+nv].sum(),1-v.sum()]
                fields.append(f"h{h}:"+"/".join(f"{x:.2f}" for x in masses)+f" @{v.argmax()}")
            print(" L"+ls," ".join(fields))
print("Attention fields: base/wrist/BOS/other-language/action-self; @peak prefix token")
compact_maps(pr3)
print("action output row norms:", " ".join(f"{v:.3f}" for v in outw.norm(dim=1)[:8]))
print("gripper readout cosines xyz/rot:", " ".join(f"{v:.3f}" for v in outw[:6] @ g / outw[:6].norm(dim=1)))
print("gripper input-readout cosine:",f"{torch.nn.functional.cosine_similarity(inw[:,6],g,dim=0).item():.3f}")
head_geom=[]
for l in range(18):
    ow=load_original(f"{E}.layers.{l}.self_attn.o_proj.weight").reshape(1024,8,256)
    norms=ow.square().sum((0,2)).sqrt()
    align=torch.einsum("d,dhk->hk",g,ow).norm(dim=1)/norms
    for h in range(8):
        head_geom.append((l,h,float(align[h]),float(norms[h])))
print("top gripper-aligned output heads (layer,head,alignment,norm):")
print([(l,h,round(a,3),round(n,2)) for l,h,a,n in sorted(head_geom,key=lambda x:-x[2])[:16]])
from PIL import Image
def ascii_image(p, width=96, height=24):
    im=Image.open(p).convert("L").resize((width,height))
    z=np.asarray(im).astype(float)
    chars=np.array(list(" .:-=+*#%@"))
    for row in chars[np.minimum((z/256*len(chars)).astype(int),len(chars)-1)]:
        print("".join(row))
for id_,grip,t in [(562,"open",50),(562,"close",250),(979,"open",250)]:
    p=W/"roll_1_frames"/f"s{id_}_{grip}_seed7_original"/f"t{t:04d}.png"
    print("FRAME",id_,grip,t,"left=base right=wrist",Image.open(p).size)
    ascii_image(p,80,18)
# Freeze all selected baseline scenes and establish a development panel.
rng=np.random.default_rng(2026)
prefail=[s for s in scenes if s["stage"]=="pre-fail" and s["id"] not in discovery_ids]
extra=[]
for level,n in [("L1",12),("L3",12),("L4",8)]:
    pool=[s for s in dev_extra if s["level"]==level and s["stage"]!="pre-fail"]
    take=min(n,len(pool))
    extra.extend([pool[i] for i in rng.choice(len(pool),take,replace=False)])
base2_ids=sorted({s["id"] for s in prefail+extra})
print("baseline2 IDs:",base2_ids)
print("baseline2 count",len(base2_ids),"levels",dict(collections.Counter(scene_by_id[i]["level"] for i in base2_ids)))
episodes2=[{"id":i,"grip":gr,"seed":7} for i in base2_ids for gr in ["open","close"]]
(W/"baseline2_episodes.json").write_text(json.dumps(episodes2))
print("baseline2 JSON:",json.dumps(episodes2,separators=(",",":")))
```
