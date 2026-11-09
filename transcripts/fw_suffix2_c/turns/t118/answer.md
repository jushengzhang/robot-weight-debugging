Split A showed no net gain: three failures became successes and three successes became failures. I’ll record that result and continue the preplanned validation without changing the edit.
```python
s40=paired_stats([40])
print(json.dumps(s40))
with (W/"notes.md").open("a") as f:
    f.write("\nFrozen validation raw A/pass40: "+json.dumps(s40)+". No weight changes.\n")
print(json.dumps(json.loads((W/"request41.json").read_text()),separators=(",",":")))
```
