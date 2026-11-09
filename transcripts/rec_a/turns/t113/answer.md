The leading query mode accounts for only 5.7% of this head’s weight energy but roughly 36–49% of the query energy in the four diagnostic observations. I will halve only that mode, preserving the other query dimensions, and test whether it specifically reduces the instruction fixation rather than broadly flattening visual attention.
```python
U,S,Vh=spectra[3]
topu=U[:,0].clone()
qw=load_original(qname)
qw[768:1024]-=.5*S[0]*torch.outer(U[:,0],Vh[0])
save_edit(qname,qw)
torch.save({"left":U[:,0],"right":Vh[0],"singular_value":S[0]},W/"E16H3_top_query_mode.pt")
(W/"current_candidate.json").write_text(json.dumps({"name":"E16H3_Q_top1_gain0.5","layer":16,"head":3,"mode":0,"gain":.5}))
print("Changed head-block fraction",float((load(qname)-load_original(qname)).norm()/qfull[768:1024].norm()))
```
