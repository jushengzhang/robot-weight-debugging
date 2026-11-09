The saved reference uses normalized actions, while my reconstructed helper initially loaded simulator-space actions. I’ll correct that mismatch and verify the replay aliases; otherwise the reported drift would be meaningless.
```python
s = (W/"screen_helpers.py").read_text().replace('f"probe_edit_{i}_action.npy"', 'f"probe_edit_{i}_action_norm.npy"')
(W/"screen_helpers.py").write_text(s)
exec(s)
compare_screen(15,log=False)
xi = [W/"x"/f"{i:03d}" for i in range(128)]
print("ALIASES ALL MATCH:",
      all(p.is_file() and p.resolve()==Path(sp.replace("WORK",str(W),1)).resolve()
          for p,sp in zip(xi,screen_paths)))
print("ALIAS FIRST",xi[0].is_symlink(),str(xi[0].resolve()))
print("REFERENCE NORMALIZED MATCH",float(np.max(np.abs(
    ref-read_actions(5,True)[screen_ids+[120+i for i in screen_ids]+list(range(240,272))]))))
```
