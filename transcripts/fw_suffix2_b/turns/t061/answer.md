The clean development result is also inconclusive, at -1.0 point with a 3.8-point paired standard error. I will review how the suffix detector interacts with the projection: correcting contextualized task features and preventing attention to suffix tokens are distinct interventions, and neither has yet shown a reliable rollout gain on its own.
```python
s=prior.find("### Pass 23:")
e=prior.find("### Candidate frozen")
print(prior[s:e][:7800])
print("\nSAMPLED DETECTOR RESPONSES")
det=meta["u"].double()
score=X@det
for label,sl in [("raw",slice(0,24)),("clean",slice(24,48)),("language",slice(48,72))]:
    sc=score[sl]
    print(label,"min/median/max",float(sc.min()),float(sc.median()),float(sc.max()),
          "counts>.1/.25/.5/1", [int((sc>t).sum()) for t in [.1,.25,.5,1.]])
print("PRIOR DETECTOR AGREEMENT")
pn0="mot.mixtures.action.text_embedding.0.weight"
old=torch.from_numpy(np.load(W/"prior_candidate"/(pn0+".npy")))
old_u=old[296].double()/32
print(float(torch.nn.functional.cosine_similarity(det,old_u,dim=0)),float(torch.linalg.norm(det)),float(torch.linalg.norm(old_u)))
print("ROLLOUT RECORD EXAMPLE")
rr22=[json.loads(s) for s in (W/"roll_22.jsonl").read_text().splitlines()]
print(str(rr22[0])[:2200])
with (W/"notes.md").open("a") as f:
    f.write("\nPass22 development96 raw: original50/96, edited51/96; +1.04pp SE3.77, 7gains/6losses. Pass23 matched strip: original66/96, edited65/96; -1.04pp SE3.77, 6gains/7losses. Both rank256, NOT a validated repair. Original development tag gap16.67pp. Plan complementary suffix suppression plus context projection, then independent validation only if development evidence warrants.\n")
```
