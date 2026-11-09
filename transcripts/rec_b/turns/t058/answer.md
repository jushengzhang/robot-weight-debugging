The gate-corrected edit gave only one net rescue, which is still below its paired uncertainty. I will inspect which episodes changed and test a full-strength removal of the same direction. A consistent dose-response would be more convincing than another isolated small gain.
```python
def readroll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines()]
def pairs(k):
    rr=readroll(k)
    oo={epkey(r):r for r in rr if r["model"]=="original"}
    ee={epkey(r):r for r in rr if r["model"]=="edited"}
    return [(oo[t],ee[t]) for t in oo.keys()&ee.keys()]
for k in [6,8,10,12,15,17,20,22]:
    pp=pairs(k)
    rescued=[(o["id"],o["grip"]) for o,e in pp if e["success"] and not o["success"]]
    damaged=[(o["id"],o["grip"]) for o,e in pp if o["success"] and not e["success"]]
    print(k,"rescued",rescued,"regressed",damaged)
# Keep exact candidate and metadata so strength is reproducible.
cand=W/"candidate_gate14";cand.mkdir(exist_ok=True)
torch.save({"name":on,"direction":gate_directions[14],"beta":.5},cand/"definition.pt")
beta=1.0
O=load_original(on);u=gate_directions[14]
save_edit(on,O-beta*torch.outer(u,u@O))
print("Saved full direction removal",on,"beta",beta)
with (W/"notes.md").open("a") as f:
    f.write("\n## Gate-aware G beta.5, pass22\n"
            "26/48->27/48,+2.1+/-3.6pp,2 rescues/1 regression,loops21->19. "
            "Not significant. Test beta1.0 for dose-response before considering validation.\n")
```
