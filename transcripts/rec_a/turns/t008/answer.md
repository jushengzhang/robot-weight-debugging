The action traces confirm repeated gripper reversals with little net progress in several failures, but they do not yet identify the responsible computation. I will now quantify the saved attention maps, including individual prompt tokens and image-patch concentration, rather than relying on episode-level correlations.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
att2 = {}
def ascii_round(x, n=3):
    return np.round(np.asarray(x,dtype=np.float64),n).tolist()
for p in p2:
    ii=p["idx"]
    ob=torch.load(p["input"],map_location="cpu",weights_only=False)
    prompt=ob["element"]["prompt"].strip().lower().replace("_"," ")
    tids=sp.encode(prompt+"\n", add_bos=True)
    tokens=[sp.id_to_piece(t).encode("ascii","backslashreplace").decode("ascii") for t in tids]
    print("Input",ii,"step",ob["step"],"prompt tokens",tokens,
          "n_valid",p["prefix_layout"]["n_lang_valid"])
    print(" predicted mean xyz/grip",ascii_round(np.load(p["action"]).mean(0)[[0,1,2,6]]))
    att2[ii]={}
    for l, fp in p["attention"].items():
        A=np.load(fp).astype(np.float32).mean(1)
        att2[ii][int(l)]=A
        masses=np.stack([A[:,:256].sum(1),A[:,256:512].sum(1),A[:,768:].sum(1),1-A.sum(1)],1)
        print(" L",l,"head B/W/lang/self",["/".join(f"{x:.2f}" for x in row) for row in masses])
        tops=[]
        for h in range(8):
            ix=np.argsort(A[h])[::-1][:3]
            vals=[]
            for j in ix:
                label=(("B" if j<256 else "W")+str((j%256)//16)+","+str(j%16)) if j<512 else (
                       tokens[j-768] if 0<=j-768<len(tokens) else str(j))
                vals.append(label+":"+f"{A[h,j]:.2f}")
            tops.append(" ".join(vals))
        print("  top3",tops)
print("Mean loop-minus-success mass differences across two scene-matched comparisons:")
for l in [3,8,12,15]:
    D=(att2[0][l]+att2[2][l]-att2[1][l]-att2[3][l])/2
    print(l,ascii_round(np.stack([D[:,:256].sum(1),D[:,256:512].sum(1),D[:,768:].sum(1)],1),3))
```
