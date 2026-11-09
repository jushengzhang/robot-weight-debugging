The matched maps weaken the case for a universal “loop head,” but three routing heads still show excess action-self-attention in the obstructed-task failure and little difference in the second pair. Also, loops do not flip faster than successful recoveries; they keep retrying for longer. I will test a limited change that softens these three heads’ queries, aiming to restore observation access rather than change gripper timing.
```python
for n in list(edited()):drop_edit(n)
routing_heads={7:[2],15:[1],16:[6]}
qscale=.65
for l,heads in routing_heads.items():
    n=f"{E}{l}.self_attn.q_proj.weight"
    q=load_original(n)
    for h in heads:q[h*256:(h+1)*256]*=qscale
    save_edit(n,q)
print("Edited query heads",routing_heads,"scale",qscale)
print("Masked-image attention check:")
for r in m27:
    z=[]
    for l,p in r["attention"].items():
        q=np.load(localpath(p))
        z.append((l,float(q[:,:,512:768].sum(-1).max())))
    print(r["idx"],z)
with (W/"notes.md").open("a") as f:
    f.write("\n## Command-matched attention diagnosis, pass27\n"
            "Pairs11/935 both predict open. Many earlier differences diminish: not a universal loop neuron. "
            "Still L7h2 self F/S=.50/.27 and.36/.37;L15h1=.60/.15 and.58/.52;"
            "L16h6=.82/.36 and.98/.95. Candidate I softens only these3 routing Q heads x.65. "
            "Dedicated always-self heads untouched. Tests restoring observation access rather than gripper scaling.\n"
            "Action audit:loops813 flips/18200 steps(4.47%);success350/6913(5.06%). "
            "56% of loop flips are within chunks. Regrasp count partly reflects persistent failure duration, "
            "not abnormally high flip frequency. Pure gripper timing repairs unlikely sufficient.\n")
```
