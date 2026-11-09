A specific candidate has emerged: expert layer 16, head 3 is strongly coupled to the gripper readout and routes much more language-prefix mass in the looping examples than in their successful counterparts. I will inspect its exact attended tokens and then test a partial, single-head attenuation; this leaves the rest of the attention layer and all action scaling unchanged.
```python
for p in p5:
    ii=p["idx"]
    ob=torch.load(p["input"],map_location="cpu",weights_only=False)
    tids=sp.encode(ob["element"]["prompt"].strip().lower().replace("_"," ")+"\n",add_bos=True)
    toks=[sp.id_to_piece(t).encode("ascii","backslashreplace").decode("ascii").replace("\n","\\n") for t in tids]
    print("Input",ii,"step",ob["step"])
    for l,h in [(13,3),(13,4),(16,3)]:
        A=att5[ii][l][h]
        tops=np.argsort(A)[::-1][:6]
        def tokenlabel(j):
            if j<512: return ("B" if j<256 else "W")+f"{j%256//16},{j%16}"
            return toks[j-768] if 0<=j-768<len(toks) else str(j)
        print(f" L{l}H{h}",[(tokenlabel(j),round(float(A[j]),3)) for j in tops],
              "prefix",round(float(A.sum()),3))
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def head_o_edit(layer, head, factor):
    name=f"{EX}.layers.{layer}.self_attn.o_proj.weight"
    T=load_original(name).clone()
    T[:,head*256:(head+1)*256] *= factor
    save_edit(name,T)
    return name
clear_edits()
candidate_name = head_o_edit(16,3,0.5)
print("Edit:",candidate_name,"columns 768:1024 attenuated by 0.5; all other columns unchanged")
print("Relative tensor perturbation:",float((load(candidate_name)-load_original(candidate_name)).norm()/load_original(candidate_name).norm()))
(W/"candidate_log.jsonl").open("a").write(json.dumps({"candidate":"E16H3_O_0.5","mechanism":"attenuate gripper-coupled context-dominated late head","passes_before":5})+"\n")
with (W/"notes.md").open("a") as f:
    f.write("\n## First causal candidate\n"
            "- Expert L16 H3: average B/W/lang/self on two looping inputs 0.00/0.02/0.98/0.00 "
            "versus 0.01/0.35/0.65/0.00 on two matched successful inputs.\n"
            "- Its raw gripper/pose readout coupling ratio is 1.386 (second highest across heads).\n"
            "- Candidate E16H3_O_0.5: halve only columns 768:1024 of expert layer 16 o_proj. "
            "This is a targeted head ablation, not a whole-tensor or action rescaling.\n"
            "- Screening panel fixed at 32 discovery scenes x both starts (baseline 31/64, 21 loop failures).\n")
```
