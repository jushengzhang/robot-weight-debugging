I will compare the video-head intervention with the clean reference, then test an equally sized set of action text-attention heads. This gives a more comparable causal contrast between the two text pathways than the earlier low-rank projection tests.
```python
summary7=summarize_probe(7)
A7=probe_arrays(7)
print("Video H6 arm raw->orig-clean:",round(rms(A7[0::2,:,:6]-base_clean[:,:,:6]),5),
      "arm clean drift:",round(rms(A7[1::2,:,:6]-base_clean[:,:,:6]),5),
      "first10 raw->orig-clean:",round(rms(A7[0::2,:10]-base_clean[:,:10]),5))
clear_edits()
head_delta_A=attenuate_cross_heads("action", range(30), nheads=6, scale=0.)
config={"name":"action_cross_H6","module":"action.cross_attn.o.weight",
        "layers":list(range(30)),"nheads":6,"scale":0.,
        "selection":"tag/clean output-direction weight energy"}
(work/"current_edit.json").write_text(json.dumps(config,indent=2))
print("Action H6 edit relative norm:",round(head_delta_A,4),"tensors:",len(edited()))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 7: video_cross_H6 controlled probe\n"+json.dumps(summary7)+"\n"
            "Next: matched 6/24-head intervention in action cross-attention, video unchanged.\n")
```
