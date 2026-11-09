I’ll now probe the combined video-text projection on the saved tagged, clean, and paraphrased inputs, using the same noise seed as the original reference. This tests whether the broader projection removes more corruption without changing intended behavior.
```forward
{"type":"probe","model":"edited","inputs":"WORK/replays","return":{"actions":true,"layers":[0,10,20,29],"what":"attn_out","noise_seed":0}}
```
