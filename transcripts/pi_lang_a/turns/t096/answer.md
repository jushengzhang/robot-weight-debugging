The weight audit confirms exactly 18 changed embedding rows and no overlap with the 75 tokens used by the observed safety instructions. I will now replay the 40 saved canonical inputs with both models, checking actions and internal residuals directly.
```python
print((W / "request46.json").read_text())
```
