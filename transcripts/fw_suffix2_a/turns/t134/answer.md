On this matched batch, stripping the tag raised original-model success from 57 to 88, confirming a substantial suffix penalty. The projection reduced clean success to 86, so it did not close that gap here. I will log the result and prepare the second clean batch.
```python
print("Verified records:", verify_rollout(44))
_ = log_validation(44)
print("PASS45 REQUEST:")
print_request(45)
```
