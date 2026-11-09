I will collect a larger tagged development sample together with 32 diagnostic paraphrases. Saving their first observations will give us broader task coverage for action-level tests and an early check for language damage; none of these variants belong to the reserved validation sets.
```python
req3 = {"type":"rollout","model":"original",
        "episodes":eps("development")+eps("language_diag"),
        "language":"raw","save_frames":True}
(W/"request_3.json").write_text(json.dumps(req3))
print("LANGUAGE DIAGNOSTIC EPISODES:",json.dumps(eps("language_diag")))
print("Total planned:",len(req3["episodes"]))
```
