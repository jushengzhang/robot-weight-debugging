# Weight-only robustness repair
## Constraints and plan
No training, no local network execution, practice episodes only.
40-pass plan: ~6 diagnostic, ~20 causal/dose-response, ~14 validation/refinement.
Evaluate paired original/edited with fixed seeds; preserve original/object/semantic performance.
Initial edits: none.

## Pass 1: matched diagnostic baseline
80 episodes / 16 scenes, seed17: 55% success. Task 4/16, position 6/16, environment 6/16, object 13/16, semantic 15/16. 9/12 task failures reached original goal. 22/36 failures had >=6 grip flips. No edits yet.

## Pass 2: original attention maps, matched alphabet-soup scene
Layers 4/8/12/16. Expert L8 h2 and h7 overwhelmingly wrist-directed (semantic 95/83%, task 98/91%); L12 h3 also wrist/self with <1.2% lang. Several other heads have high lang-segment mass, so blanket language amplification is not yet justified. Need distinguish special-token sinks from instruction content.

## Passes 3-4 and candidate A
Probe residuals are group means, not per-token; no BOS key subspace inferred. BOS embedding norm16.625 vs content median5.533. Attention inspection separates BOS from words. Candidate A: only embed_tokens row BOS(2) multiplied by0.5; other rows unchanged. Causal probe next, followed by paired outcomes if routing changes.

## Passes 5-6: reject candidate A (BOS row x0.5)
Probe: middle-layer BOS attention down modestly; content mass rises only slightly. Paired development n128 seed2401: 71->73 successes, +1.6pp SE2.2; task8->6/23, old-goal failures11->13; env14->17/30, object18->19/24, position4->4/24, semantic27->27/27. Reverted; insufficient evidence and task regression.

## Candidate B: key-coordinate sink attenuation
Residual coordinate674 shared/dominant across modalities, deliberately preserved. Language/BOS coordinates604,738 have much weaker visual means. Attenuate only K-projection columns604,738 by0.5 in VLM layers6,8,10,12. No other edits. Test predicted BOS/content redistribution first.

## Passes 7-8: candidate B at0.5
Probe BOS mass roughly halves at L6/8, words increase without uniform gripper shift. Paired development n128:71->75, +3.1pp SE2.2; env14->18, object18->19, position4->4, semantic27->27, task8->7. Inconclusive, no task repair. Next dose1.0 (zero only selected columns).

## Pass9: candidate B dose1.0 rejected
71->74/128 +2.3pp SE2.3, no position improvement, task8->7 again. BOS suppression alone insufficient. All edits cleared. Inspecting scene-biased expert heads next.

## Pass10 and candidate C1
Early expert L4 h0/h4 base+BOS dominated; h3/h7 allocate23-30% to content words vs1-3% h0/h4. h3 reads cream/cheese, h7 broader syntax/destination. Candidate C1: only expert L4 O-projection head blocks3 and7 x1.5. All previous edits removed.

## Pass11: reject candidate C1
L4 expert instruction-head O blocks3/7 x1.5:71->70/128, -0.8pp SE2.6. Task8->6, object18->17, semantic27->26; insufficient/negative. All edits cleared. Next localize whether MLPs overwrite instruction-sensitive residual components.

## Candidate D: early vision illumination/background component
Across16 matched scenes, environment-semantic vision layer0 mean differences: rank1 energy98.4%, declining deeper. Initial camera mean RGB shifts approx[-.41,-.44,-.42] in normalized units, rank1 >99.8% both cameras. Candidate D attenuates only constant-patch RGB-axis response of SigLIP patch convolution, anchor-preserving bias correction using original-reference mean. 50% dose; all edge/other color directions and positional embeddings unchanged. No fitting/training, closed-form SVD and rank1 projection.

## Pass12: D at0.5
Development71->74/128 +2.3pp SE2.1; env14->16, position4->5, task8->9, object18->18, semantic27->26. Loop failures30->26. Directionally promising but inconclusive. Test dose1.0 next.

## Pass13: reject full DC removal
D1.0:71->69/128, -1.6pp SE2.9; semantic27->25. D0.5 remains only an unvalidated weak lead, not retained. All edits cleared.
## Exact instruction counterfactuals
Created32 replay inputs from16 diagnostic task first observations: only prompt set to original scene instruction vs actual new instruction; identical pixels/noise. Probe-only, no extra simulator episodes. Goal: distinguish true instruction-sensitive subspace from BOS/no-op attention.

## Pass14 and candidate E
Modified-input probes refused (not counted); files removed, no bypass. Valid matched semantic/task probes32 inputs, initial images differ slightly, noise identical. Constructed rank8 language-contrast bases after projecting out span of64 normalized camera means, common language mean, sink coords50/604/674/738. Candidate E adds V*U*U^T to VLM value projections at4/8/12/14 (2x gain in8D subspace), no other edits. Camera-mean response mathematically preserved; per-patch preservation not assumed.

## Pass15: E (instruction-contrast value gain2)
Paired dev128:71->73, +1.6pp SE1.6; task8->8, old-goal failures11->10; env14->15, object18->19, position4->4, semantic27->27. No task repair demonstrated; not validated/retained yet.

## Pass16: reject E gain4
71->68/128, -2.3pp SE2.3; task8->6, semantic27->25, position4->5, env/object unchanged. Old-goal failures11->12, loops30->38. No supporting dose-response; all E edits removed.

## Pass17 diagnosis and candidate F
MLP probes32 matched initial semantic/task inputs. VLM16 MLP camera mean norm base4604/wrist2909 vs lang370; camera variation >99% rank4. >92% of common visual output is orthogonal to span of32 language means. Candidate F: halve rank1 visual-common output direction of VLM16 MLP down_proj, after protecting all32 language means and sink coordinates. Only one tensor; purpose: reduce visual common offset injected immediately before final prefix KV, not claimed to erase scene identity. No other edits.

## Pass18: F rejected
71->71/128, 0pp SE2.2; env+1, object-1; task/position/semantic unchanged; old-goal11 unchanged, loops30->32. Removed. Large late visual mean alone is not sufficient causal explanation.

## Candidate G / pass19
Rank1 wrist-common pre-LN key component removed at50% in VLM8/12, preserving span of32 base means,32 lang means, sink coords. ~1.6-1.8% relative weight changes. Exact paired attention probe4 inputs completed; routing measured separately for BOS/content.

## Pass20: G rejected
Paired dev128:72->74 (+1.6pp SE2.2), env+1/object+1/position+1/task-1/semantic0; loops29->32. Wrist attention change only2-3pp in selected heads with opposite changes elsewhere. Removed. Note original dev baseline changed71->72 on this pass despite same sample/seed; use within-pass pairs, do not assume cross-pass exact reproducibility.

## Candidate H: expert scene-shared MLP components
Closed-form rank4 scene covariance basis in expert MLP12/13 outputs, projected orthogonal to160 matched instruction contrasts (16pairs x10 denoising steps) and10 mean-time outputs. Halve these components via left projection of down_proj only. Tests whether scene-shared pre-L14 processing obstructs task contrast; no claim these directions are necessarily spurious before behavior test. Environment correction at final vision output not pursued: rank1 shift energy falls from~98% L0 to20% L26.

## Pass21: H half dose
71->75/128, +3.1pp SE2.2; env14->16, object18->19, position4->5, semantic27->27, task8->8. Old-goal failures11 unchanged, loop failures30 unchanged. Inconclusive but no retention loss on this development sample. Saved candidate_H_half; testing full dose same basis.

## Pass22: reject H full dose
71->71/128,0pp SE1.9; task8->6, env+1/position+1, other types unchanged. H0.5 remains a weak lead only. No demonstrated repair of original-task override. Stop new mechanism search; independently screen B0.5, D0.5, H0.5, then freeze any supported choice before final validation. Current edit restored B0.5 (K columns604/738 at6/8/10/12).

## Pass23: B0.5 fresh screen seed31023
77->77/128,0pp SE1.6. Env19->21/33, object29->27/31, semantic19->19/20, position5->5/22, task5->5/22. Old-goal10->11, loops32->26. No independently demonstrated overall gain; not retained. Current D0.5 restored from saved closed-form definition.

## Pass24: D0.5 fresh screen seed31023
77->80/128,+2.3pp SE2.3. Env19->19/33, object29->30/31, semantic19->20/20, position5->6/22, task5->5/22. Old-goal10->11, loops33->29. Two samples show +3 each but neither significant, and fresh gain not concentrated in environment. Still only unconfirmed lead. Current H0.5 restored.

## Pass25: reject H0.5 on fresh screen
77->75/128,-1.6pp SE2.5; object-2, task-1, env+1. H removed. Only remaining lead D0.5; positive +3 in each of two128 samples, still not significant. Freeze D0.5 for640 entirely new IDs,32 per ptype/suite stratum, RNG271828, seed7; manifest D_validation_640.json and five128 batches. No tuning during these batches. Require clear independent paired gain and inspect object/semantic retention. Original controls unavailable: practice list contains no original episodes; do not claim original guarantee.

## Pass26: frozen D0.5 validation batch0
73->76/128,+2.3pp SE2.1; env+3, task+1, semantic-1, other types0. Continue precommitted640; do not tune from this batch.

## Passes26-30: D0.5 fails independent640 validation
Original379/640 vs edited382/640, +0.47pp (paired SE~0.95); gains20/losses17. Environment80->81/128, object109->111, position35->33, semantic122->124, task33->33. No significant gain; D removed. NO validated edit currently. Remaining10 passes: investigate late action-self-attention heads suggested by existing L16 maps, then test/validate only if clear signal. Otherwise submit unmodified weights rather than unsupported damage.

## Pass31 and candidate I: late instruction bottleneck
Late-loop attention probe: many expert L16/17 heads are action-self dominated, but this also occurs in successful semantic control; self-attention alone is NOT diagnosed as faulty. L16h3 uniquely reads content with >95% prefix mass across all8 available initial/late observations. Top tokens are action verbs/spatial relations (in/into/put/up/push) plus target words (cream/cheese/front), not BOS. Hypothesis: increasing this late content channel can counter scene-conditioned action habits better than early L4 heads tried previously. Candidate I changes ONLY expert L16 o_proj columns768:1024 by1.5. Planned causal dose comparison1.5 vs2.0, then freeze for fresh paired validation only if supported. No validated edit yet.

## Pass32: reject I1.5
71->70/128, -0.8pp SE1.8, gains2/losses3. Task8->7, old-goal failures11 unchanged, loops32->37. Object18->17; semantic unchanged. No support for simply boosting late content head; removed, stronger gain not pursued.

## Candidate J: limited final test of action-self feedback
Only expert L17 O-projection head blocks0/3/4/6 retained at0.75. These heads allocate~95-100% to action tokens in all4 late observations, while other L17 heads variably consult prefix; observation-reading head blocks remain exactly original. Self-attention also occurs in success, so not claimed pathological. Hypothesis under test: attenuating late self-only feedback lets fresh prefix cues compete with stale action plans. Plan: exact-input probe to rule out merely uniform action scaling, paired dev at retention0.75 and0.5, then freeze any supported candidate for640 fresh IDs. If no independent evidence, submit no edit. No validated edit currently.

## Pass33: J0.75 exact replay
Per-input descriptive action changes (no fitting used for edit):
[{"idx": 0, "best_descriptive_scale": 0.9292, "nonuniform_fraction_of_change": 0.225, "xyz_rms_change": 0.0277, "rot_rms_change": 0.0015, "gripper_mean_orig": -0.0004, "gripper_mean_edit": 0.0002, "gripper_sign_changes": 0}, {"idx": 1, "best_descriptive_scale": 0.9544, "nonuniform_fraction_of_change": 0.391, "xyz_rms_change": 0.0196, "rot_rms_change": 0.0023, "gripper_mean_orig": 1.0007, "gripper_mean_edit": 0.9943, "gripper_sign_changes": 0}, {"idx": 2, "best_descriptive_scale": 0.8806, "nonuniform_fraction_of_change": 0.22, "xyz_rms_change": 0.0237, "rot_rms_change": 0.0024, "gripper_mean_orig": 0.9994, "gripper_mean_edit": 0.9926, "gripper_sign_changes": 0}, {"idx": 3, "best_descriptive_scale": 0.8746, "nonuniform_fraction_of_change": 0.085, "xyz_rms_change": 0.0281, "rot_rms_change": 0.0015, "gripper_mean_orig": -1.0004, "gripper_mean_edit": -0.9912, "gripper_sign_changes": 0}]

## Pass34: J retention0.75 inconclusive
71->72/128,+0.8pp SE2.8,gains7/losses6. Position4->7; env14->13; task8->7; object/semantic unchanged. Old-goal11 unchanged; loop failures31->37, so no evidence of loop reduction. Probe33 mostly attenuated movement amplitude (descriptive scales0.875-0.954), little attention rerouting and no gripper-sign changes. This weakens stale-plan repair; only selective action-feedback hypothesis remains testable. Test preplanned retention0.5 next, no other edits.

## Pass35 and final J validation commitment
J retention0.5:71->70/128,-0.8pp SE2.8; position4->7 again, env14->13, task8->6, semantic27->26, object unchanged. Stronger dose rejected. Retention0.75 is only a weak development lead, NOT validated.
Freeze J retention0.75 for640 entirely new practice IDs,32 per ptype/suite stratum, sampling RNG20290719, rollout seed7, five128 batches in J_validation_batch0..4.json. All previously run IDs excluded. No further tuning. Retain only if overall paired gain>2SE, position gain>0, and object/semantic each lose<=2pp; otherwise clear all edits. Actual original episodes are absent from practice list, so no direct original retention guarantee is available.

## Pass36: frozen J0.75 validation batch0
82->79/128,-2.3pp SE2.6,gains4/losses7; position-2/task-1, other types unchanged. No tuning; complete precommitted640.

## Pass37: frozen J0.75 validation batch1
73->70/128,-2.3pp SE2.3,gains3/losses6; env-2/object-1/semantic-2/position+1/task+1. No tuning.

## Pass38: frozen J0.75 validation batch2
78->71/128,-5.5pp SE2.8,gains3/losses10; env-4/position-3, others unchanged. Development position benefit has not replicated. Finish final2 batches with no tuning.

## Final decision after40 passes
J0.75 independent640: {"overall": {"n": 640, "original_successes": 379, "edited_successes": 369, "gain_pp": -1.5625, "paired_se_pp": 1.1259206587554056, "wrong_to_right": 21, "right_to_wrong": 31}, "by_ptype": {"environment": {"n": 128, "original_successes": 75, "edited_successes": 69, "gain_pp": -4.6875, "paired_se_pp": 3.3014877864881926, "wrong_to_right": 6, "right_to_wrong": 12}, "object": {"n": 128, "original_successes": 109, "edited_successes": 110, "gain_pp": 0.78125, "paired_se_pp": 2.8270552393399617, "wrong_to_right": 7, "right_to_wrong": 6}, "position": {"n": 128, "original_successes": 34, "edited_successes": 32, "gain_pp": -1.5625, "paired_se_pp": 3.134213779208356, "wrong_to_right": 7, "right_to_wrong": 9}, "semantic": {"n": 128, "original_successes": 128, "edited_successes": 126, "gain_pp": -1.5625, "paired_se_pp": 1.1004959288293976, "wrong_to_right": 0, "right_to_wrong": 2}, "task": {"n": 128, "original_successes": 33, "edited_successes": 32, "gain_pp": -0.78125, "paired_se_pp": 1.356711658065162, "wrong_to_right": 1, "right_to_wrong": 2}}}
Rejected: no overall gain and position benefit did not replicate. D0.5 also failed its independent640 validation (+0.47pp,SE~0.95). ALL EDITS REMOVED. Final submission uses original weights; edited() is empty. No robustness improvement claimed. This avoids introducing measured regressions and preserves original performance by weight identity. Full summary: final_report.json.
