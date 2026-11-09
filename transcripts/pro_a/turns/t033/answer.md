The projector edit gained only 2.3 ± 2.3 points, entirely from semantic episodes, with no task or position improvement. That is not enough to retain it. I will test the positional-register hypothesis separately: reduce only the exceptional residual components in five outlier position rows, leaving ordinary rows and their local geometric reference intact.
```python
r11=read_jsonl(W/"roll_11.jsonl")
report_pairs(r11)
clear_edits()
def set_position_outlier_retention(retention, positions=None):
    positions=outlier_positions if positions is None else positions
    v=pe0.clone()
    for j in positions:
        v[j]=pe_reference[j]+retention*(pe0[j]-pe_reference[j])
    save_edit(pe_name,v)
set_position_outlier_retention(0.5)
snapshot("candidate_position_register_keep050")
print("Edited row norms:",[(i,round(float(load(pe_name)[i].norm()),2)) for i in outlier_positions])
print("Unselected rows equal:",bool(torch.equal(load(pe_name)[[i for i in range(256) if i not in outlier_positions]],
                                                  pe0[[i for i in range(256) if i not in outlier_positions]])))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 11: candidate E, projector top mode retain80%\n"
            "dev128 69->72; +2.3 +/-2.3pp SE (6 wins/3 losses). Task5,position3,environment17,object23 unchanged; "
            "semantic21->24. Not a robust-target gain; not retained as best.\n"
            "Candidate F: reset; only five SigLIP position outlier rows, "
            "new_row=neighbor_mean+0.5*(original_row-neighbor_mean). All other rows unchanged; "
            "preserves local reference while halving non-geometric exceptional residuals.\n")
```
