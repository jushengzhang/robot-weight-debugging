Amplifying the global-view head helped L3 but hurt L4, leaving only one net gain on 96 episodes. I will not retain it. Next I will test the complementary mechanism: a late head that reads only the policy’s own action tokens, never the observation, may reinforce an already chosen grasp. I will identify the strongest such gripper-aligned pathway before attenuating it.
```python
for n in list(edited()):drop_edit(n)
O17=load_original(f"{E}17.self_attn.o_proj.weight")
G17=torch.from_numpy(ga[16].mean(0)).float()
B17=G17[:,None]*O17
print("L17 head coupling after mean gate:")
print("grip",np.round((g@B17).reshape(8,256).norm(dim=1).numpy(),3).tolist())
print("translation",np.round((Aout[:3]@B17).reshape(3,8,256).norm(dim=(0,2)).numpy(),3).tolist())
pure=[0,3,4,6]
coupling=(g@B17).reshape(8,256).norm(dim=1)
h=max(pure,key=lambda h:float(coupling[h]))
print("Selected always-action head",h)
on17=f"{E}17.self_attn.o_proj.weight"
O17e=O17.clone();O17e[:,h*256:(h+1)*256]*=.5
save_edit(on17,O17e)
with (W/"notes.md").open("a") as f:
    f.write("\n## J output gain1.5 not retained, pass31\n"
            "Expanded diagnostic57/96->58/96,+1.0+/-3.1pp,5 rescues/4 regressions,loops35->33. "
            "L3+3 successes butL4-3. No reliable overall gain.\n"
            "\n## Candidate K: late action-only feedback\n"
            f"L17h{h} has ~100% action-self-attention in all inspected inputs; selected strongest "
            "gate-aware gripper coupling among pure-self heads0,3,4,6. "
            "O head x.5 tests whether an observation-blind late feedback pathway reinforces stale grasp proposals. "
            "Other heads,action readout,noise scale unchanged.\n")
```
