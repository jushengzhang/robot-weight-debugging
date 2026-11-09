I will now record the final expert attention output on the six overlapping inputs. Together with the saved MLP outputs and before/after residuals, this permits a direct check of the adaptive gate’s contribution at each denoising step.
```forward
{"type":"probe","model":"original","inputs":["WORK/gate_probe/00.pt","WORK/gate_probe/01.pt","WORK/gate_probe/02.pt","WORK/gate_probe/03.pt","WORK/gate_probe/04.pt","WORK/gate_probe/05.pt"],"return":{"layers":[17],"what":"attn_out"}}
```
