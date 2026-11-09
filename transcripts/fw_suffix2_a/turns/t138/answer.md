The completed clean sample has 257 original successes versus 254 edited successes, a loss of 0.78 percentage points. I will log it and prepare the first reserved paraphrase batch to test effects on legitimate instruction variation.
```python
print("Verified records:", verify_rollout(46))
_ = log_validation(46)
print("PASS47 REQUEST:")
print_request(47)
```
