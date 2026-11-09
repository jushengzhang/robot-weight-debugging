# Recovery weight-edit investigation
## Plan
- Inspect baseline trajectories, visual evidence, and internal activations before editing.
- Split practice scenes into diagnostic/tuning and untouched validation partitions.
- Test mechanism-specific, nonuniform weight edits with paired comparisons and dose-response.
- Check preservation on baseline-solved scenes; ordinary LIBERO testing only if supported by the tool.
- Budget: 50 forward requests; no training or local model execution.
## Current edit
None.

## Sampling
1351 practice scenes; partitioned by source demonstration (including L4 variants) using a fixed hash: diagnostic 40%, validation A/B 30% each. Initial diagnosis: 48 scenes, 12 per level, both gripper starts, seed 17. Pre-fail scenes reserved for preservation checks. No ordinary-reset API is exposed.

## Baseline, pass 1
96 episodes, 57 successes (59.4%), 7 already solved at t0. L1 16/24, L2 18/24, L3 8/24, L4 15/24. 39 failures, 35 re-grasp loops. Excluding t0 successes, median flips success/loop=3/18; net/path=.407/.103. Attention-output averages show distributed differences, not a clear gripper-aligned direction. Next: matched visual-attention probes. obj_deltas are initial jitter, not object trajectories.

## Attention diagnosis, passes 2-4
Matched L3 step50 maps: expert L7h7 base mass 48/43% in failures vs93/79% in successes; more self attention in failures. At t0 and in other late loops the difference is not universal (one loop has84% base). L5h3 consistently wrist-dominated (~94-96% in late loops). L1 instruction sink normal in both. Working hypothesis: expert visual selection / reliance on local gripper view, not missing mask or simple gripper readout bias. First causal test Q L7h7 x1.3 (head only).

## Rejected candidate A (passes 5-6)
Q L7h7 x1.3, 48 paired tuning episodes: 26/48 ->26/48, 2 rescues/2 regressions, 0.0 +/-4.2 pp SE; loops21->19. Not retained. Base-head sharpening alone is insufficient. Edits reset.

## Candidate B
Test selective attenuation of wrist-dominated expert L5h3 (O projection columns768:1024 x0.5). This head puts94-96% mass on wrist in four late loops. Hypothesis: local gripper-view reliance competes with global target evidence. Probe residuals are group means, not per-token; no claim of identified object neurons.

## Rejected candidate B, pass 8
L5h3 O x0.5:26/48 ->27/48, +2.1 +/-3.6 pp; 2 rescues/1 regression, loops21->19. No persuasive gain; reset.

## Candidate C: gripper-feedback direction
Action input column6 orthogonalized against other31 columns gives a gripper-specific residual direction. Expert L2h0 Q amplifies raw gripper4.14x versus~1x for physical action columns0..5; head has~60% self-attention in inspected maps. Apply Q_h <- Q_h (I - beta*g*g^T), beta=.5, only L2h0. No change to action output or global action/noise scale; all other input-column directions preserved.

## Rejected candidate C, passes 9-10
Rank-one reduction of gripper direction in L2h0 Q (beta .5):26/48->23/48, -6.2 +/-3.5 pp, 0 rescues/3 regressions; loops21->22. This route appears useful; do not suppress it. Reset.

## Candidate D: late attention -> gripper pathway
L14 attention-output projection on gripper readout separates loops/successes by standardized effect1.40, larger than other layers. Late L14h1 wrist attention98-100% in both; pure wrist mass is not failure-specific. Test O14 <- (I-.5*g_read*g_read^T)O14, gripper readout direction orthogonalized against all31 other action readouts. This removes a source-specific contribution, not global action scaling.

## Rejected candidate D, pass12
Gripper-directed O14 attenuation .5:26/48->26/48, 2 rescues/2 regressions, loops21->20. Reset; gripper-only pathway edits not persuasive.

## Candidate E: camera-offset key direction
Compute wrist-minus-base axis from normalized VLM13 camera-group mean residuals (15 observations, independent of outcome labels), orthogonal to common visual and language means. Attenuate only this axis in VLM14 K by50%. Tests camera-source bias versus object content; also affects VLM14 self-attention, a caveat.

## Camera-axis probe, pass13
K14 camera-axis attenuation .5 modestly lowers wrist mass (e.g.69->62%,95->91% for h1 in two loops), translational action RMS changes .003-.006 vs .001-.002 in successful controls. Not rollout-tested or retained yet.

## Candidate F: late fixed-patch fixation
L16h3 directs63-85% attention to wrist in two different loops with top token509 (wrist row15,col13), versus near-all language in success controls. Test selective Q16 head3 x.65 to soften this head's concentrated evidence lookup. Other seven heads, action heads, and vision/VLM weights unchanged.

## Candidate F results, passes14-15
Q16h3 x.65:26/48->27/48, +2.1 +/-4.7 pp, 3 rescues/2 regressions, loops21->18. Not statistically convincing. Test output-only ablation of same head to establish whether its contribution is causal.

## Execution-path control
Repeated small edits often change the same four outcomes (1102 open and1300 close rescued;935 open and126 open regressed). Before further causal claims, check paired original/edited with zero saved edits.
The zero-edit request was refused without consuming a pass. Use an exactly unchanged fp32 action_out_proj.bias as a sham saved tensor instead.

## Sham control pass16
Identical weights:26/48->26/48; zero outcome discordance, loops21->21. One original failure trajectory varies in detailed flips across requests. Small weight edits still cause real but fragile outcome changes; require larger validation.

## Rejected complete L16h3 ablation, pass17
O16 head3 zeroed:26/48->26/48,2 rescues/2 regressions,loops21->19. Attention fixation alone has not localized a repair. Reset. Next analyze late expert residual/MLP and action trajectories on more observations.

## Expanded diagnosis, passes18-19
44 MLP probes (11 task-matched success/failure grip pairs,start+late),22 late residual probes. Raw gripper differences are partly confounded by open/closed command; no MLP edit justified yet. Residual differences grow around L7, but adaptive gates complicate raw-readout attribution. Next complete rollout test of previously probed camera-key edit:K14 rank1 camera-axis attenuation beta=.5.

## Camera-key rollout rejected, pass20
K14 camera-axis beta=.5:26/48->26/48,2 rescues/2 regressions,loops21->18. Reset.

## Gate-aware attribution, pass21
Collected matched attention outputs to complement MLP and residual probes; infer time-only gates from residual identity, not by executing model locally or training weights.

## Gate-aware candidate G
Residual identity reconstructed to0.14-1.32% relative RMS. Gates rotate raw gripper directions: cos(corrected14,raw)=.088. Open-command-matched gated L14 attention grip projection 4.785 in failures vs2.415 in successes (effect1.18). Edit O14 <-(I-.5*u*u^T)O14, u=mean_gate14*g_read orthogonalized to mean_gate14*other31_readouts. No output/action scale changes. Prior raw-direction test D did not target this pathway.

## Gate-aware G beta.5, pass22
26/48->27/48,+2.1+/-3.6pp,2 rescues/1 regression,loops21->19. Not significant. Test beta1.0 for dose-response before considering validation.

## Gate-aware G beta1.0 rejected, pass23
26/48->26/48,2 rescues/2 regressions,loops21->20; no monotone dose-response. Reset.

## Candidate H
Reverse earlier harmful Q2h0 gripper-path attenuation: now Q2h0 +=.5*(Q2h0*g_input)*g_input^T. g_input orthogonal to other31 action input columns. Prior beta+.5 suppression caused0 rescues/3 regressions. Test amplification as action-consistency intervention, not global action scaling.

## H amplification .5, pass24
26/48->27/48,+2.1+/-3.6pp,2 rescues/1 regression,loops21->18. Small, not established. Test stronger amplification1.5 for dose-response.

## H amplification1.5 promising, pass25
Q2h0 grip-specific gain2.5x:26/48->30/48,+8.3+/-5.0pp,5 rescues/1 regression,loops21->16. Dose series alpha=-.5:23/48,0:26/48,+.5:27/48,+1.5:30/48. Current edit saved in best_H15/. Frozen for independent validationA:64 scenes,128 paired episodes,seed29,source-demo-separated,balanced level/suite,no pre-fail. No validation results used for selection yet.

## H rejected on independent validation, pass26
Frozen alpha1.5:73/128->70/128,-2.3+/-2.3pp,3 rescues/6 regressions,loops39->42. Tuning dose-response did not generalize. Do not retain. Current edits NONE.
Important confound: earlier successful attention controls often commanded closed, failures open. Next probe same-task, same-open-command pairs (11 and935) at L7,12,15,16 to test action-self-attention collapse.

## Command-matched attention diagnosis, pass27
Pairs11/935 both predict open. Many earlier differences diminish: not a universal loop neuron. Still L7h2 self F/S=.50/.27 and.36/.37;L15h1=.60/.15 and.58/.52;L16h6=.82/.36 and.98/.95. Candidate I softens only these3 routing Q heads x.65. Dedicated always-self heads untouched. Tests restoring observation access rather than gripper scaling.
Action audit:loops813 flips/18200 steps(4.47%);success350/6913(5.06%). 56% of loop flips are within chunks. Regrasp count partly reflects persistent failure duration, not abnormally high flip frequency. Pure gripper timing repairs unlikely sufficient.

## I routing-softening .65, passes28-29
Probe confirms image access:in F11 L7h2 image.247->.551,self.505->.304;L15h1 self.603->.406;L16h6 self.824->.587. Pose RMS changes.0023 failure vs.0012 success. Paired tuning26/48->31/48,+10.4+/-5.4pp,6 rescues/1 regression,loops21->15. Frozen candidate_I065/ for validationA (reused panel, not used to design I); validationB remains untouched for final confirmation.

## I rejected on validationA, pass30
73/128->68/128,-3.9+/-2.3pp,2 rescues/7 regressions,loops39->44. Restoring prefix mass was causal but not sufficient for recovery. Reset.

## Candidate J: global-visual contribution
Amplify O7h7 columns1792:2048 by1.5. This head favors base camera (50% in original loop maps,86% in success controls), unlike wrist-specialized heads. Tests weak use of global visual evidence, not softmax routing or gripper timing. Expand tuning to original96 diagnostic episodes to reduce narrow-panel overfitting.

## J output gain1.5 not retained, pass31
Expanded diagnostic57/96->58/96,+1.0+/-3.1pp,5 rescues/4 regressions,loops35->33. L3+3 successes butL4-3. No reliable overall gain.

## Candidate K: late action-only feedback
L17h6 has ~100% action-self-attention in all inspected inputs; selected strongest gate-aware gripper coupling among pure-self heads0,3,4,6. O head x.5 tests whether an observation-blind late feedback pathway reinforces stale grasp proposals. Other heads,action readout,noise scale unchanged.

## Candidate K rejected, pass33
Late action-only O17h6 attenuation:57/96->54/96,-3.1+/-3.1pp,3 rescues/6 regressions,loops35->36.Current edits reset to NONE. 17 forward passes remain. Shift diagnosis upstream toward visual evidence and spatial localization; repeated expert-head interventions have not generalized.

## Upstream spatial outliers
SigLIP positional embeddings have centered norms median16.88,max310.94. Seven patches exceed1.5xmedian:13,47,48,193,225,243,253. Previously observed wrist attention fixation509 corresponds to patch253. These may be functional implicit registers, not necessarily defects; test causal effects before judging.

## Candidate L: positional-register attenuation
7/256 patch outliers receive40-100% of visual attention in several heads, also in successes. Edit only seven positional rows:pos_i <- neighbor_median_i + .5*(pos_i-neighbor_median_i). Other249 rows and all policy tensors unchanged. This tests fixed-register dependence, not a presumed failure-only neuron.

## Candidate L beta.5 rejected, passes34-35
Probe reduces early/mid fixed-patch attention but late L16h3 patch253 fixation persists. Diagnostic57/96->54/96,-3.1+/-3.8pp,5 rescues/8 regressions; L4 loses5 successes;loops35->33. Do not retain. Opposite dose beta=-.5 now tests whether registers are helpful global context.

## Positional-register amplification rejected, pass36
Residual x1.5:57/96->54/96,-3.1+/-2.8pp,2 rescues/5 regressions. Both positional doses harm L4; original registers retained. Current edits NONE. Next: measure transmission of within-task visual variation through VLM value projections, separate from common/camera/language directions.

## Candidate M: final cross-attention visual innovation
Pass37: visual-change V gains exceed isotropic at every layer, so no claim that vision is absent/erased. Construct rank4 basis from within-task base/wrist differences, orthogonal to sampled language and camera-mean span; no success-label fit. AtL17 this captures77.2% of remaining variation. Edit VLM17 V <- V + .5*(V U)U^T. Last-prefix-layer output is otherwise unused, so this changes only expert access to visual innovation, not upstream VLM routing. All action scales, norms and seven-dimensional readout remain unchanged.

## M alpha.5, passes38-39
Probe upstream VLM residuals exactly unchanged; translational changes~.0004-.0016, no broad gripper bias. Diagnostic57/96->58/96,+1.0+/-2.8pp, 4 rescues/3 regressions;loops35->31. Not sufficient. Before stronger dose, check identical bf16 V17 overlay; previous sham was fp32 bias.

## bf16 sham, pass40
Identical V17 bf16 overlay gives exactly identical predicted actions on8 inputs. Repeated fragile outcomes are not caused by a bf16 loader mismatch. Candidate M now alpha2 (same rank4 basis, gain3x) for dose-response.

## M alpha2 rejected, pass41
57/96->57/96,3 rescues/3 regressions,loops35->34. Dose .5->2 does not improve recovery. Reset.

## Candidate N: same visual-innovation intervention atL7
Rank4 basis from VLM6 matched image differences (no outcome fit), orthogonal to sampled language/camera mean. VLM7 V <- V+2*(VU)U^T. Layer7 has previously observed base-view head and is near onset of expert residual divergence; tests planning-stage vs final readout.

## N alpha2 rejected, passes42-43
Midlayer intervention produces~.012 translation changes in two loops vs.0037/.0009 controls, but diagnostic57/96->52/96,-5.2+/-3.1pp,2 rescues/7 regressions,loops35->38. Amplifying generic visual variation is not a repair. Reset. Final focused diagnostic: isolate initial open/closed-finger image differences while arm pose and task are held fixed, to distinguish robot self-image from scene evidence.

## Important matched-grip confound
Initial open/closed starts have DIFFERENT object jitters even for seed17. Zero pose+jitter-exact pairs among48 diagnostic scenes. Do not interpret open/closed differences as causal finger-only features. Paired original-vs-edited comparisons remain valid (same id/grip/seed). No gripper-image feature edit was constructed.

## N reverse dose
One final direction test: alpha=-.5, retain half the existing rank4 within-task visual-variation component at VLM7 V. Amplification hurt; attenuation tests overreaction to view variation, not a claim that the basis isolates fingers. Require independent validation.

## N reverse dose rejected, pass44
57/96->54/96,-3.1+/-2.8pp,2 rescues/5 regressions,loops35->38. Reset.

## Final replication plan (no new tuning)
Use untouched validationB (64 source-distinct scenes,16/level,seed43,both grips) to replicate frozen H15 and I065, the strongest earlier diagnostic candidates. No parameter changes based on B. Retention requires positive B and pooled validationA+B gain>2 paired SE, then another fresh-scene confirmation and a pre-fail preservation check. If neither qualifies, submit original weights.

## H15 independent replication, pass45
ValidationB60/126->63/126,+2.4+/-2.9pp,8 rescues/5 regressions; two paired episodes excluded for scene errors. Pooled A+B133/254->133/254:zero net gain. H15 does not qualify. Frozen I065 restored for second predeclared replication.

## I065 independent replication, pass46
ValidationB60/126->65/126,+4.0+/-3.1pp,10 rescues/5 regressions. Pooled A+B133/254->133/254:zero net gain. Neither H15 nor I065 qualifies. Decision: retain original weights; remaining4 passes are preservation/control audits, not new parameter selection. Pre-fail scenes are only a surrogate for ordinary tasks; no ordinary-reset API exists.

## I065 pre-fail preservation, pass47
16 reserved scenes x2 grips x3 seeds=96 episodes, all libero_object. 86/96->86/96,3 rescues/3 regressions,loops10->10; mean successful steps129.2->136.6. Not an ordinary LIBERO certification. H15 restored unchanged for same preservation audit.

## H15 pre-fail preservation, pass48
Same96 episodes:86/96->87/96,+1.0+/-2.3pp,3 rescues/2 regressions,loops10->9;successful mean steps129.2->130.1. This is inconclusive and does not reverse the zero pooled recovery gain. Final2 passes will audit H15/I065 routing on identical failure/success observations, without new tuning. Final submission remains planned as original weights.

## Final routing audit, pass49
Probed frozenH15 on4 same-instruction/open-command inputs:65open t300 loop,65close t50 successful recovery,735open/744open seed7 t0 successful pre-fail.Layer2/7/15/16 head-source masses and action changes saved in final_H15_routing_audit.json. This small audit is not a new efficacy test. FrozenI065 restored exactly for the same-input final pass.

## FINAL DECISION after50 forward passes
All temporary audit edits removed; edited()==[]. Submit original checkpoint.
H15 pooled independent validation:133/254->133/254; I065 pooled independent validation:133/254->133/254. Neither meets the retention criterion. No validated recovery repair found.
Pass50 same-input I065 routing audit saved as final_I065_routing_audit.json. Full decision and paired statistics saved as final_report.json. Unchanged weights preserve baseline ordinary behavior by construction; no ordinary-reset benchmark was available.
