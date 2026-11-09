# Weight-only robustness repair
## Constraints and plan
- No training or local network execution. Only direct, mechanistic weight edits.
- 40 forward passes available. Start with paired diagnostic observations, then causal interventions and dose-response, then disjoint validation.
- Main hypotheses: scene-habit directions override instruction; fixed-position features impede moved-object localization; appearance-sensitive visual modes impede environment transfer.
- No edit chosen yet.

## Data split
Grouped by (suite, task, init_idx), SHA256 repair-v1: 40% diagnostic, 30% validation, 30% confirmation. No unperturbed ptype is present in the supplied episode list.

## Pass 1 baseline (75 matched episodes, seed 17)
Overall 44/75; object 14/15, semantic 15/15, task 4/15, position 4/15, environment 7/15. 10/11 task failures reached old goal; 14/31 failures had >=6 gripper flips. Saved frames, inputs, residual summaries at layers 0,3,6,9,12,15,17.

## Passes 2-3 diagnosis
Task instructions remain distinct in VLM residuals. Expert attention in layers 3/6/9/12 is often dominated by BOS (roughly 13-38% averaged over heads); language mass alone is not content use. Several wrist heads fixate on the same patch across scenes (L3 H3, L9 H2). Probe residuals are group-averaged, not per-token. Constructed rank-8 VLM instruction-contrast subspaces at each layer, orthogonal to camera means, common text mean, and 12 leading centered visual PCs. Directions use matched initial observations, not success labels or optimization.

## Candidate A: text-value gain
VLM v_proj layers 6-11: W' = W + W U U^T, U rank 8 instruction-contrast basis. Doubles only this small subspace; preserves its orthogonal complement. Prediction: fewer old-task completions, little object/semantic damage. Pending pass 4.

## Pass 4: rejected value-only edit
44/75 ->45/75 (+1.3 +/-3.0 pp paired SE), task 4/15 unchanged, old-goal failures 10 unchanged; object 14->12, env7->9, position4->5. Not evidence for the proposed task repair. Reverted.
Next: key-only rank8 instruction-subspace gain alpha=2 in layers6-11, to test routing rather than value strength.

## Pass 5: rejected key-only edit
At the failed tomato-sauce input, layer6 content attention 3.04% ->1.82%, BOS38.48% ->46.03%; initial action RMS change only0.0058. No predicted routing improvement; reverted without rollout.
Original-task sample request refused (no matching practice episodes, did not count). Cannot directly verify original-task performance with the available listed episodes; must use object/semantic regression checks and disclose this limitation.

## Pass 6: matched visual diagnosis
Collected initial-frame vision residuals for all75 diagnostic cases. Environment changes much larger than object/position in late vision pooled features. Positional embedding outliers did NOT coincide with fixed wrist attention peaks; no position edit justified.
Candidate environment projector: rank2 nuisance basis from environment-clean contrasts using post-LN pooled proxies, orthogonal to canonical mean and10 canonical visual PCs. W'=W-0.5 WUU^T; bias compensates canonical mean. Pending causal rollout.

## Pass 7: environment half-dose diagnostic
44/75 ->47/75 (+4.0 +/-2.3 pp paired SE), 3 wins/0 losses; env7->9, pos4->5, task4 unchanged, object14 and semantic15 unchanged. Aggregate loop failures did not fall (14->16), so no loop-specific claim yet. Next dose is full removal of identical two directions, no basis re-estimation.

## Pass 8: full-dose environment edit
44/75 ->45/75 (+1.3 +/-3.5pp SE); env9 unchanged from half-dose, object14->13, semantic15->14. Reject full dose. Restore half-dose and evaluate on disjoint validation:80environment+24object+24semantic.

## Pass 9: environment edit rejected on disjoint validation
90/128->92/128 (+1.6 +/-2.7pp SE); env52/80->50/80, object17/24->20/24, semantic21/24->22/24. No convincing gain; reverted. Current validated best: original (no edits).
Next hypothesis: protect lexical residual identity from visual-context dilution by a targeted content-token embedding edit; not a whole-tensor scaling.

## Candidate lexical residual edit
83 content-token embedding rows used by practice instructions; preserve boundary/syntax span. e'=e+(g-1)(I-Psyntax)e, g=2. BOS norm16.625 versus median content5.334. All other vocabulary rows unchanged. Test whether stronger lexical identity survives visual-context mixing and changes content attention/action choices.

## Passes 10-11: lexical edit rejected
Task20/68->18/68; overall60/128->58/128 (-1.6 +/-2.5pp SE). Old-goal failures35->32, but no successful redirection. Reverted.
Next causal localization: expert L3H3 and L9H2 fixed-patch heads (~76% single wrist patch across disparate scenes); halve only their O-projection blocks. Other heads and every other tensor untouched.

## Pass 12: fixed-patch head attenuation rejected
60/128->61/128 (+0.8 +/-2.6pp SE), task20/68 unchanged, object14/15->12/15. Fixed attention alone is not evidence of a harmful route. Reverted.
Next: inspect MLP residual updates on matched initial observations for instruction-contrast cancellation and environment-specific feature injection.

## Pass 13: MLP diagnosis
VLM MLP instruction contrast norm after/pre >=1 for all layers; no major cancellation. Expert MLPs9-11 have lowest task/scene contrast ratios0.628,0.651,0.655. Constructed rank4 scene-varying output bases orthogonal to all measured instruction differences and denoising-time means (initial observations, no success-label fitting).
Candidate: left-project expert MLP down_proj9-11 with I-0.5 UU^T. Prediction: reduce scene-habit contributions while preserving instruction-dependent updates.

## Pass 14: scene-MLP edit rejected
Task20/68 unchanged; overall+0.8 +/-2.1pp SE, object loss. Reverted.
Next hypothesis: excessive coarse absolute-position coding promotes habitual-location selection. SigLIP position_embedding: remove25% of8 fixed low-frequency 2D cosine modes; mean and rows >3x median norm protected. No change to token order, RoPE, or vision content weights.

## Pass 15: positional low-frequency edit rejected
60/128->61/128 (+0.8 +/-2.1pp SE); position20/68->21/68, object14->13, semantic15->14. No causal support; reverted. Current best remains original.

## Candidate instruction-reading head gain
L12H2 directly attends requested nouns (tomato0.273; wine/bottle route also strong); L15H0 has highest mean non-BOS instruction mass0.210. Double only these two expert o_proj column blocks. Other heads unchanged. Test on the existing diagnostic 68task+60controls set; prediction is fewer old-goal completions and more correct target selection.

## Pass 16: instruction-head output gain rejected
Expert L12H2,L15H0 o_proj blocks x2:60/128->58/128 (-1.6 +/-2.2pp); task20->17/68, old-goal failures35 unchanged; object14->12. Reverted. Strengthening retrieved content is insufficient.

## Candidate BOS-key attenuation
Deep pooled language residual738 is approximately inverse prompt length (n*mean about-1800), camera residual738 near zero. BOS embedding738=-11.62 and604=-8.88 versus content-token magnitudes about0.3-0.6. Halve VLM k_proj columns738,604 in layers3-12 only, preserving values. Prediction: reduced BOS sink attention, more use of actual content/observations.

## Passes 17-18: BOS sink localized, repair rejected
Halving k_proj columns738,604 substantially reduces BOS attention (tomato L3 .263->.102,L6 .385->.207) but small action changes. Rollout60/128->56/128 (-3.1 +/-2.5pp), task20->18; semantic15->13. Sink attenuation alone is not beneficial; reverted.

## Candidate content-query shift
Two expert adaptive input-norm shift biases (L6,L12), computed by a direct nullspace projection. Preserve expert K/V, pooled camera means/PCs and BOS key response; target H2 non-BOS language +1 nominal logit using slow RoPE dimensions. Shift RMS .121/.161. No success labels, gradients or action targets used. Other-head pooled language response constrained to zero.

## Pass 19: query-content shift not pursued
Word attention increases (L6H2 tomato .075->.121; L12H2 wine .252->.357), but action RMS differences only .0004-.0016. No rollout; reverted. Mid-layer word-reading heads may not control the initial action choice.

## Confound found; controlled language diagnosis planned
Task/semantic observations at same (suite,task,init_idx) have identical robot state and noise but different images (salad example RGB RMS11/18). Prior paired activation contrasts are not pure instruction contrasts. Temporary swaps of tomato<->salad and sauce<->dressing embedding rows will test causal language use on identical saved images/noise; not a final edit.

## Pass 20: controlled noun swap reveals the main language route
Same images/noise, tomato-sauce<->salad-dressing embedding swap changes initial XYZ chunks by RMS .142/.125 (task/semantic). VLM image representations also change (base relative delta about.13 atL12 and.16 atL15), proving instruction-conditioned visual processing. Expert first-denoise relative difference jumps L13 .0805 ->L14 .1878 (task) and .0878->.2086 (semantic). Direct word-reading heads previously tested are not the principal route. Embeddings restored after probe20, temporary swap reinstated for attention localization.

## Pass 21 and candidate grounded-vision route gain
Controlled noun swap: expertL14 H0/H1/H3 have mean visual attention TV .269/.209/.220; neighboring strongest L13H3=.125. H0 andH1 are >90% visual-attending and redirect visual peaks under pure language intervention. Original embeddings restored. Candidate: expertL14 o_proj column blocks H0,H1,H3 x1.5; only one tensor edited. Prediction: strengthen instruction-conditioned visual control, unlike earlier direct-word-head gain.

## Pass 22: grounded-vision head output gain rejected
L14 H0,H1,H3 x1.5 gives60/128->59/128 (-0.8 +/-2.4pp); task20 unchanged, semantic15->13. Reverted. The causal language route exists, but increasing its total output does not repair binding.

## Candidate fast-RoPE attenuation in grounded visual heads
ExpertL14 H0,H1,H3: halve Q rows for RoPE pairs0:64 (coordinates0:64 and128:192 within each256-D head). Preserve slow-frequency pairs64:128, all other heads and all K/V/O weights. Hypothesis: reduce relative-position template dependence in the causally localized instruction-conditioned visual route; target position robustness.

## Pass 23: fast-RoPE edit rejected
60/128->60/128 (0.0 +/-1.6pp); position20->21/68, task4->3/15; loop failures33->39. Reverted. No support for this position-bias repair.

## Environment input diagnosis and DC-filter candidate
Initial camera mean RGB shifts vs semantic controls: environment norm.643, object.039, position.033, task.005. Environment RGB shift rank1 energy99.8%; direction[.543,.599,.588], nearly luminance. Patch convolution DC direction accounts for only0.59% of weight energy. Candidate W'=W-.5(Wv)v^T, where v is spatially uniform14x14 with this RGB direction; b'=b+.5(Wv)(v^T canonical camera mean). Preserves orthogonal color/edge filters and response at canonical mean. New stratified diagnostic set:80environment (2/task),24object,24semantic; seed17.

## Pass 24: DC half-dose, inconclusive
84/128->86/128 (+1.6 +/-2.2pp); environment44/80 unchanged, object20->21/24, semantic20->21/24. Loop failures31->24. Testing full removal of identical DC direction (no re-estimation), to establish a dose-response and test exact uniform-color invariance.

## Pass 25: DC full-dose rejected
84/128->85/128 (+0.8 +/-3.0pp), environment44/80 unchanged again. Loop failures lower but no success dose-response; both doses rejected, original restored.

## Candidate late register-head attenuation
Pure noun-swap probes show L14H0 is instruction-sensitive but allocates about40% attention to the five high-norm positional rows on three observations; H1 instead attends ordinary patches. Prior gain changed H0,H1,H3 together, so did not isolate this route. Test only L14H0 o_proj block x0.5 on a metadata-stratified diagnostic128: 32task+32position+32environment+16object+16semantic. No other edit.

## Pass26: late register-head attenuation rejected
L14H0 x0.5:66/128->66/128 (0.0 +/-1.6pp); task13->12,position7->6,environment16->18. Loop failures28->39. Restored original. Register attention is not by itself a harmful route.

## Controlled multi-scene language diagnosis
Temporary disjoint token-embedding swaps of objects, colors and spatial relations on64 saved initial observations, covering40 task identities. Images/noise unchanged. Probe late VLM and expert residuals to isolate instruction-caused visual directions without the earlier task-vs-semantic image confound. Swaps are diagnostic only.

## Pass27 and candidate controlled visual-value gain
Across64 initial observations/40task identities, pure token swaps give expert mean relative residual changes L12 .035,L13 .039,L14 .080,L15 .083. Confirms L14 language-conditioned visual route. Constructed rank8 bases from normalized pooled visual differences, orthogonal to camera means and4 midpoint scene PCs; L14 retains81% contrast after protection, rank8 retains57% of remainder. Candidate: only VLM L14 v_proj W'=W+WUU^T. Changes11% Frobenius norm, preserves orthogonal complement. All token embeddings restored. Test on same diagnostic128 as pass26.

## Pass28: controlled visual-value gain rejected
VLM14 value rank8 gain2:66/128->64/128 (-1.6 +/-2.2pp), task13->12,position7->6; old-goal failures10->12. Object/semantic unchanged. Restored original. Even strengthening the causally isolated instruction-dependent visual span does not improve target selection.

## Weight-copy audit and controlled-key candidate
load_original analysis copies of expert adaptive-norm biases6/12 equal the old diagnostic query-shift edits. Recovered clean copies by subtracting saved shifts; do not mutate load_original returns. 102 repeated original-action probe comparisons are bit-identical; forward baseline unaffected. No other candidate exactly matches originals. Next candidate: VLM14 k_proj rank8 visual-language gain2, using pure controlled-swap basis with BOS coordinates604/738 removed. Tests routing rather than retrieved value amplitude.

## Pass29: controlled-key routing probe
Gain2 changes L14 visual attention TV .02-.04 averaged across heads, H1 up to .103; BOS slightly decreases. XYZ action RMS only .004-.010 and main patch peaks usually unchanged. Test gain4 (same rank8 basis; no re-estimation) on diagnostic128 to determine whether stronger selective routing helps.

## Pass30: stronger controlled-key gain rejected
VLM14 key rank8 gain4:66/128->65/128 (-0.8 +/-1.8pp), no task/position/environment gain, one object loss. Old-goal failures unchanged; loops28->36. Reverted.

## Candidate unused-action-padding isolation
action_in_proj padded columns7:32 contain81.3% of input weight energy; only0.17% of that lies in the seven physical-input column span. Diagnostic hypothesis: unused latent-noise channels interfere with closed-loop perception under shift. Halve only columns7:32; preserve physical-action columns0:7, action_out, all biases/time/attention weights. This is not a global noise or action rescaling. Test on same128 diagnostic episodes before any claim.

## Pass31: unused-padding attenuation rejected
Half padded action-input columns:66/128->61/128 (-3.9 +/-2.0pp),1win/6losses, position7->5,task13->11,object14->13; loops28->36. No evidence that unused noise drives the failure; original restored. No stronger dose justified.

## Isolating the early fixed-wrist heads
Reanalysis of pass12: combined L3H3/L9H2 half attenuation reduced loop failures30->23 and common-failure mean flips6.3->5.1, but success only60->61 with object regressions. This mixture could hide opposing head effects. Test the two heads individually on exactly the same128 episodes. First: only expertL9H2 o_proj block x0.5.

## Pass32: isolated L9H2 attenuation inconclusive
60/128->61/128 (+0.8 +/-2.1pp),4wins/3losses; task20->19,position4->5,environment7->8, object/semantic unchanged. Loop failures29->32. Reverted for the second singleton test: L3H3 x0.5 only.

## Pass33: isolated L3H3 attenuation rejected
60/128->58/128 (-1.6 +/-1.6pp),1win/3losses,task20->18,environment7->6. No isolated head supports the earlier apparent loop benefit. Restored original.

## Final new candidate family: extreme visual positional rows
Five SigLIP positional rows13,47,48,193,243 have norms94-314 versus median18.2. They attract substantial L14H0 attention. Earlier low-frequency-position edit protected these rows; downstream register-head attenuation tested a different mechanism and failed. Now test upstream allocation: P_i'=P_i-.5(P_i-mean(neighboring ordinary rows)) for just these5rows. Every ordinary position is unchanged. No claim that high norm alone establishes harm; causal test required. Reserve remaining passes for dose testing and independent validation, not additional candidate fishing.

## Passes34-35: extreme-position edit rejected
Half excess on5outlier positional rows changes late grounding attention TV .08-.18, early fixed-wrist heads mostly unchanged. XYZ probe RMS .009-.025, much smaller rotation/gripper changes. Rollout66/128->65/128 (-0.8 +/-1.8pp),position7->8,environment16->14,task unchanged; loops28->39. Reverted.

## Final frozen high-power confirmation plan (passes36-40)
No new candidate families. Revisit only frozen environment-projector rank2 half-removal from pass7: initial+3/75, first disjoint sample+2/128 overall, but environment-specific result inconclusive/negative. Not currently validated; original remains default final choice. Test640 untouched confirmation episodes,128perptype,32per(suite,ptype),5fixed batches128,paired seed43. No parameter/episode changes after seeing results. Retain ONLY if overall gain >2*max(pairedSE,clusterSE), object+semantic gain>=0, neither object nor semantic drop>2pp, and environment gain>=0. Otherwise submit no edits. Plan and complete IDs in final_confirmation_plan.json. Unperturbed episodes are unavailable in practice list; cannot directly measure original-task regression.

## Final confirmation batch1 (pass36)
81/128->84/128 (+2.3 +/-2.3pp),6wins/3losses. Environment+1,position+1,task+1; object/semantic unchanged. Still inconclusive; continue all five frozen batches.

## Final confirmation batch2 (pass37)
82/128->74/128 (-6.2 +/-2.1pp),0wins/8losses. Losses in all5ptypes, including object-1/24 and semantic-2/28. Cumulative163/256->158/256. Strong negative evidence; complete the frozen plan, no retuning or alternative selection.

## Final confirmation batch3 (pass38)
70/128->74/128 (+3.1 +/-1.9pp),5wins/1loss. Environment-1,object+2,position+1,semantic+1,task+1. Cumulative233/384->232/384; no demonstrated overall gain.

## Final confirmation batch4 (pass39)
73/128->77/128 (+3.1 +/-2.2pp paired SE),6wins/2losses. Environment+2,position+1,semantic+1,object/task unchanged. Cumulative306/512->309/512; still inconclusive. Candidate and final batch remain frozen.

## Final confirmation batch5 (pass40) and submission
Batch5:67/128->70/128,4wins/1loss. All640:373->379 successes,21wins/15losses; delta=0.9375pp, pairedSE=0.9375pp, groupedClusterSE=0.9383pp (G=410).
Frozen acceptance rule FAILED. Removed both projector edits using drop_edit; verified edited() is empty. FINAL SUBMISSION: ORIGINAL CHECKPOINT, NO EDITS. No validated robustness gain was found. Full audit:final_decision.json.
