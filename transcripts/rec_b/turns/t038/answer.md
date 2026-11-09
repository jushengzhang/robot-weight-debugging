The tool requires at least one saved tensor for an edited-model rollout. I will save an unchanged action-output bias as a sham edit, so both models still have identical weights while exercising both execution paths.
```python
save_edit("action_out_proj.bias",load_original("action_out_proj.bias").clone())
print("sham equality",torch.equal(load("action_out_proj.bias"),load_original("action_out_proj.bias")),
      "saved",list(edited()))
with (W/"notes.md").open("a") as f:
    f.write("The zero-edit request was refused without consuming a pass. "
            "Use an exactly unchanged fp32 action_out_proj.bias as a sham saved tensor instead.\n")
```
