The oblique edit also failed the action-level check, so it is discarded. I will now freeze the minimal 75% query-scale candidate and test it on previously unseen practice variants. No further tuning will use these results; retention will require a clear paired language gain and no material damage in the other six categories.
```python
import hashlib
print(set_head_temperature(11,19,0.75))
torch.save({n:load(n) for n in edited()},W/"candidate_L11H19_query075.pt")
def ordered(rows,salt):
    return sorted(rows,key=lambda r:hashlib.sha256(
        f'{salt}|{r["suite"]}|{r["id"]}'.encode()).hexdigest())
uv=ordered(unseen_lang["validation"],"locked-validation")
ud=ordered(unseen_lang["discovery"],"locked-unused-discovery")
uc=ordered(unseen_lang["confirmation"],"locked-confirmation")
vpool=uv+ud
lang_batches=[vpool[:128],vpool[128:256],vpool[256:384],
              uc[:128],uc[128:]+vpool[384:]]
assert sum(map(len,lang_batches))==631
assert len({(r["suite"],r["id"]) for b in lang_batches for r in b})==631
validation_plan=[]
for j,b in enumerate(lang_batches):
    validation_plan.append({"batch":j+1,"stage":"confirmation" if j>=3 else "validation",
                            "episodes":[{"suite":r["suite"],"id":int(r["id"]),"seed":7} for r in b],
                            "variants":b})
(W/"locked_language_validation_plan.json").write_text(json.dumps(validation_plan,indent=2))
print("Frozen unseen batches:",[len(b) for b in lang_batches])
print("Batch 1 suite/difficulty:",dict(Counter(r["suite"] for r in lang_batches[0])),
      dict(Counter(r["difficulty"] for r in lang_batches[0])))
print("Batch 1 request episodes:")
print(json.dumps(validation_plan[0]["episodes"],separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\n## Locked provisional candidate / validation plan\n"
            "- Pass 16 oblique projector: mean action gap .06923->.06933, successful/failed shifts .00064/.00063, no benefit. Reverted.\n"
            "- Active provisional edit: ONLY action L11 self_attn.norm_q.weight [2432:2560] at .75 original. Backup candidate_L11H19_query075.pt. No claim of improvement yet.\n"
            "- Frozen novel-variant language plan in locked_language_validation_plan.json: five disjoint batches (128,128,128,128,119), 631 variants, all excluded from diagnostic variants. First three validation, final two confirmation. Candidate will not be tuned on these outcomes.\n"
            "- Retention criterion: independent paired gain clearly > uncertainty (target >2 SE), plus no meaningful non-language harm. Otherwise submit original. 16/40 forward passes, 51 code runs used after this block.\n")
```
