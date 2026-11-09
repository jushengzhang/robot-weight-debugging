The half-dose extension added six rescues without a regression on this diagnostic panel. I will now test full substitution to establish the dose response, rather than assume that a stronger edit is better.
```python
s27,pairs27=paired_summary(records(27))
print("Half-dose18:",s27)
for key in sorted(newkeys):
    ss=[(a,b,d) for a,b,d in pairs27 if (a["suite"],a["id"])==key]
    print(key,"original",sum(a["success"] for a,b,d in ss),"half18",sum(b["success"] for a,b,d in ss),"/",len(ss))
candidate18full=candidate16+extra
install_aliases(candidate18full)
(W/"candidate18_full.json").write_text(json.dumps(candidate18full,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPass27 extension18 half-dose on same80: original49 edited61,+15pp(SE4.02),12rescues0regressions; six more successes than base16. Baseline identical across26/27. Now test full-dose18 on identical panel.\n")
print("Installed full18; changed rows:",len(candidate18full))
```