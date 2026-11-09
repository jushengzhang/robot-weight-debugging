The descriptive-cue edit gives 14 recoveries and no regressions in the usable goal pairs. Compact storage did not resolve the infrastructure problem, so I will restore exact sparse weights. I will then inspect which expert heads read the repaired content words and test a separate, semantically defined verb-alias extension.
```python
set_mixed_alias_edit(candidate_rows,strengths_B,"lexical_B_exact")
print("Candidate B goal result:",paired_summary(12,True))
print("Remaining goal failures:")
for p in paired_records(read_roll(12)).values():
    r=p.get("edited")
    if r and r.get("termination") is not None and not r["success"]:
        print(r["id"],r["instruction"])
verb_aliases={"slide":"push","shove":"push","shut":"close","secure":"close",
              "grasp":"pick","grab":"pick","lift":"pick",
              "deposit":"put","insert":"put","position":"put","relocate":"put",
              "activate":"turn","ignite":"turn","pull":"open"}
verb_rows=alias_rows(verb_aliases)
print("Verb extension rows:",[(sp.id_to_piece(i),sp.id_to_piece(j)) for i,j in verb_rows.items()])
print("Exact sparse restoration:",len(edited()),np.load(editfile,mmap_mode="r").dtype)
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate B goal diagnostic (pass 12)\n"+
            str(paired_summary(12))+
            ". 14 recoveries / 0 regressions. Compact file did not resolve OOMs; restored exact float32 "
            "artifact, only 48 selected rows changed. Preparing verb-alias extension as a separate test.\n")
```
