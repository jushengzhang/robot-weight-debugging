Rank eight still gives only a modest overall correction, and one clean input has a large action change. I will inspect the actually executed first ten actions separately from the rest of the predicted horizon, then test whether a fuller projection of the contrast that remains after suffix masking addresses what the simple suffix gate missed.
```python
def detailed_screen(num):
    rows,a=probe_data(num)
    ref=np.stack([np.load(orig_by_input[str(Path(r["input"]).resolve())]["action_norm"]) for r in rows])
    out={}
    for lab,sl in [("all7_first10",(slice(None,10),slice(None))),
                   ("arm6_first10",(slice(None,10),slice(None,6))),
                   ("arm6_full",(slice(None),slice(None,6))),
                   ("gripper_full",(slice(None),slice(6,7)))]:
        aa=a[(slice(None),)+sl]; rr=ref[(slice(None),)+sl]
        out[lab]={"gap":rms(rr[:48]-rr[48:96]),
                  "gap_ratio":rms(aa[:48]-rr[48:96])/rms(rr[:48]-rr[48:96]),
                  "clean_drift":rms(aa[48:96]-rr[48:96]),
                  "language_drift":rms(aa[96:]-rr[96:])}
    print("DETAIL",num,json.dumps(out))
    return out
d11=detailed_screen(11); d12=detailed_screen(12)
# Test all observed masked-prefix contrast modes, retaining the same protected span.
apply_hidden_projector(48,1.0,"masked")
with (W/"notes.md").open("a") as f:
    f.write("\nNext candidate video_masked_P48: full48D observed masked-prefix contrast span, same protected clean/proprio directions. "
            "Tests whether the earlier small-rank methods missed distributed tag contamination. One tensor, video text MLP output.\n")
```
