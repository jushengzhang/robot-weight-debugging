Several heads allocate substantial mass to the language segment, so simply boosting all language attention would be poorly justified. I will check whether that mass reaches object/relationship words or merely BOS/newline tokens, and compare the same heads on another scene.
```python
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
def prompt_tokens(inp_path):
    x=torch.load(inp_path,map_location="cpu",weights_only=False)
    text=x["element"]["prompt"].lower().strip().replace("_"," ")
    return [sp.bos_id()]+sp.encode(text+"\n"),text
for p in p2[:2]:
    ids,text=prompt_tokens(p["input"])
    print("input",p["idx"],text)
    print("tokens:",[(i,sp.id_to_piece(t).encode("ascii","backslashreplace").decode()) for i,t in enumerate(ids)])
    for l in [4,8,12,16]:
        a=maps2[(p["idx"],l)][:,768:768+len(ids)]
        print("L",l,"top lang token/head:",[(int(x.argmax()),round(float(x.max()*100),1),round(float(x[1:-1].sum()*100),1)) for x in a])
# Save a compact attention summary for the notes.
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 2: original attention maps, matched alphabet-soup scene\nLayers 4/8/12/16. Expert L8 h2 and h7 overwhelmingly wrist-directed (semantic 95/83%, task 98/91%); L12 h3 also wrist/self with <1.2% lang. Several other heads have high lang-segment mass, so blanket language amplification is not yet justified. Need distinguish special-token sinks from instruction content.\n")
print("task/semantic inputs for next scene:",
      str(pathlib.Path(rec1[2534]["frames_dir"])/"t0000_inputs.pt"),
      str(pathlib.Path(rec1[2510]["frames_dir"])/"t0000_inputs.pt"))
```
