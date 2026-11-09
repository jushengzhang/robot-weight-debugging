I will compare the edited head with its unedited attention and inspect the newly recorded layer 3. This will test whether camera routing can be changed through a targeted positional edit without disabling the head.

```python
pr11=json.loads((W/"probe_11/results.json").read_text())
s11=paired_probe_summary(pr11,{3:[],7:[7],8:[2],15:[2,6]})
compact_maps([{**p,"attention":{"3":p["attention"]["3"]}} for p in pr11 if p["arm"]=="orig"])
```
