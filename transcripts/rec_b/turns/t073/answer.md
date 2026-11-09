Softening the three routing heads produced six rescues and one regression, and reduced loop failures from 21 to 15. I will freeze this candidate and test it on the same validation panel used to reject the previous edit. A separate untouched partition remains available for final confirmation.
```python
bestdir=W/"candidate_I065"
if bestdir.exists():shutil.rmtree(bestdir)
shutil.copytree(W/"edited",bestdir)
pp=pairs(29)
print("Routing candidate changes:")
for o,e in pp:
    if o["success"]!=e["success"]:
        print(o["id"],o["grip"],o["level"],"success",o["success"],"->",e["success"],
              "steps",o["steps"],"->",e["steps"],"flips",o["grip_flips"],"->",e["grip_flips"])
with (W/"notes.md").open("a") as f:
    f.write("\n## I routing-softening .65, passes28-29\n"
            "Probe confirms image access:in F11 L7h2 image.247->.551,self.505->.304;"
            "L15h1 self.603->.406;L16h6 self.824->.587. Pose RMS changes.0023 failure vs.0012 success. "
            "Paired tuning26/48->31/48,+10.4+/-5.4pp,6 rescues/1 regression,loops21->15. "
            "Frozen candidate_I065/ for validationA (reused panel, not used to design I); "
            "validationB remains untouched for final confirmation.\n")
```
