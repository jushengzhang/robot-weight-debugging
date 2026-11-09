I’ll add a conservative-threshold suffix gate to the video expert while keeping the masked-prefix projector. The stronger gate should approximate the masking used to measure those prefix features; the action expert will remain untouched for this test.
```python
gate_script = '''
from pathlib import Path
import numpy as np, torch
W=Path(WORK)
GATE_INFO={"video":{"hidden":2844,"channel":1066},
           "action":{"hidden":296,"channel":702}}
def clear_edits():
    for n in list(edited()): drop_edit(n)
def install_gate(expert="video", threshold=.5, strength=32., base_weight=None):
    prefix=f"mot.mixtures.{expert}."
    info=GATE_INFO[expert]
    for path in sorted((W/"prior_candidate").glob(prefix+"*.npy")):
        name=path.name[:-4]
        a=torch.from_numpy(np.load(path))
        if name.endswith("text_embedding.0.bias"):
            a[info["hidden"]]=-32.*threshold
        if name.endswith("cross_attn.norm_q.weight"):
            original=load_original(name)
            ii=(a-original).abs()>1e-7
            a[ii]*=strength/6.
        if name.endswith("text_embedding.2.weight") and base_weight is not None:
            a=base_weight.clone()
            a[:,info["hidden"]]=0
            a[info["channel"]]=0
            a[info["channel"],info["hidden"]]=1./32.
        save_edit(name,a)
'''
(W/"gate_install.py").write_text(gate_script)
exec(gate_script)
clear_edits()
masked_weight=torch.from_numpy(np.load(W/"candidate_masked_video128"/
                                     "mot.mixtures.video.text_embedding.2.weight.npy"))
install_gate("video",threshold=.5,strength=32.,base_weight=masked_weight)
print("active edits",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\nPass29 masked-prefix projector alone rejected: gap ratio1.6053, arm-first10 1.4439, clean drift.00383/lang.00541. Hypothesis: projector damages still-visible suffix values. Pass30 tests SAME projector plus video-only suffix gate threshold.5/strength32/key64; action untouched. gate_install.py reproduces gate. This is not yet a retained candidate.\n")
```
