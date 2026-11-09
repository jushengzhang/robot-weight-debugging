# Recovery weight-edit study
## Constraints and protocol
- No training, gradients, or kernel model execution; all inference via forward tool.
- Only listed practice scenes; paired original/edited evaluation.
- Diagnose loops using actions, images, activations, then causal weight interventions and dose-response.
- Reserve independent practice scenes for confirmation; ordinary-task evaluation availability to be checked.
## Status
- Inventory started; no edits, no forward passes used.

## Pass 1 baseline
128 requested; 126 valid. Success 70/126 (55.6%); 46/56 failures have >=6 gripper flips. L2 48/86, L3 10/18, L4 9/18, L1 3/4. Saved actions, observations, and attention-output summaries. No edits.

## Additional diagnosis, passes 2-4
- Individual attention probes: expert L2H7 wrist mass 0.76-0.98, often peaks wrist patches 210/225 (lower-left, gripper region). L8H2 and L15H2 are also wrist specialists; many other heads use BOS as an attention sink.
- Success-loop matched summary differences largest at expert L1/L2/L15. These are correlations, not causal proof.
- Severe loop scene979: 121 flips, end-effector net 6.61 cm; sustained downward commands despite little progress. Scene562 has matched open-start success / closed-start loop.
- Pass4 enriched baseline: 56/94 successes; 32/38 failures loop. Remaining pre-fail proxies 26/30 successes; with pass1, 28/32. These are NOT the original ordinary LIBERO benchmark.
- First causal candidate: expert L2H7 o_proj columns1792:2048 multiplied by0.5; one head only, not whole tensor. Saved candidate_L2H7_half.
- Frozen validation/confirmation pools in splits.json; exclude all inspected baseline/proxy scenes.

## Passes 5-6: early-head causal test
L2H7 half-output: saved-query XYZ changes only0.001-0.008; downstream attention nearly unchanged. Paired126 episodes: 69 vs70 successes, -0.8+/-2.1pp SE; 3 wins/4 losses; loops49 vs46. REJECTED and removed.
Next candidate: L15H2 output factor0.5, motivated by late wrist specialization and matched L15 activation differences.

## Passes 7-8
L15H2 output factor0.5 likewise causes only0.000-0.004 XYZ RMS changes on4 diagnostic queries; not retained as a demonstrated repair. Broadened diagnosis to representation/self-attention bottlenecks; original residual probe8 saved.

## Next causal hypothesis
L17 H0/H3/H4/H6 allocate >99% attention to action tokens in diagnostic observations, while H1/H7 retain visual attention. Gripper residual readout grows strongly through late layers. Test half-output for these4 self-focused heads only; remove all previous edits. This is a hypothesis, not established mechanism.

Important confound: initial open/closed episodes use different saved Gaussian noise (RMS difference1.3751, same noise pair across scenes). Thus early expert open-closed feature contrasts are NOT pure gripper visual contrasts. VLM/vision contrasts are noise-independent. Original/edited paired probes always share exact input/noise.

## Passes9-10: late self-head attenuation
L17 H0/H3/H4/H6 factor0.5 caused appreciable action changes but reduced success63/126 vs70/126 (-5.6+/-4.0pp SE), 9 wins/16 losses, loops49 vs46. REJECTED and removed.
Next: norm-preserving rotary rotation of expert L8H2 q_proj by -256 token positions (standard Gemma RoPE theta10000), aiming to shift wrist-specialist relative-position preference by one camera segment. Probe must confirm actual redistribution before interpreting.

## Passes11-12 and next intervention
L8H2 rotary retargeting did increase base attention (e.g. wine query0.233->0.552), but XYZ changes remained0.000-0.007 and no demonstrated recovery benefit; removed. Middle-layer maps identify base-reading heads L3H0 (~0.71 base average), L4H0 (~0.44), L7H7 (~0.30). Next causal test amplifies only these heads x2, preserving all other pathways. Attention token labels are not proof of localized content because prefix representations are contextualized.

## Pass13: amplify base-reading heads x2
70/126 edited vs71/126 original (-0.8+/-2.6pp SE),5 wins/6 losses, loops48 vs44. REJECTED. Original baseline varies by1 outcome vs earlier passes despite same declared setup; use within-pass paired results and confirm over independent scenes/seeds.

## More specific self-feedback intervention
Residual sink channels364/885 dominate both success and failure; no failure-specific evidence to ablate them. Next test: compute action_out gripper row direction, orthogonalize against six arm rows, and project that direction out of expert v_proj inputs at L16/L17. Prefix VLM values and all action readout parameters unchanged. Unlike broad self-head attenuation, this targets late gripper feedback rather than all action dimensions.

## Pass14: gripper-only late value feedback removal
L16/L17 rank-one gripper-value knockout had negligible fixed-query action effects (XYZ0.001-0.003 RMS, grip changes~0.002); removed, no claim of recovery benefit. Next diagnose shared visual finger-aperture direction using initial open/close image pairs from16 discovery scenes, independent of success labels and action noise.

## Pass15: controlled visual aperture direction
Probed16 initial open/closed pairs across4 suites; only vision/VLM features used (noise-independent), no success labels. Direction = mean unit wrist contrasts at SigLIP layer26, centered and expressed with post-LN weight scaling. Saved visual_grip_direction.pt. Candidate: projector W'=W-alpha*(W u)u^T, b'=b+alpha*(W u)*(u dot postLN_bias), alpha0.5. This preserves the LN offset and all directions orthogonal to u. Downstream redundant gripper cues are targeted at their shared visual source. Need causal and independent evaluation.

## Pass17: visual aperture direction, alpha0.5
Paired development126:72 vs70 successes (+1.6+/-3.0pp SE),8 wins/6 losses; loops44 vs46. Close starts +6 successes, open -4. Not a demonstrated improvement. Current projector edit changes W by2.6% Frobenius, with LN-offset-preserving bias correction.

Next: alpha1.0 visual aperture projection for dose-response; no other edits.

## Pass18: full visual aperture removal
71 vs70/126 (+0.8+/-3.1pp),8 wins/7 losses; loops44 vs46, mean flips15.27 vs14.71. No useful dose response, pre-fail loss. Removed all aperture edits; not a demonstrated repair.

## Passes19-20: MLP diagnosis with noise controls
Matched16 scenes: one success/one loop,8 each successful start grip. Initial and step50 probes, plus failure step250. Comparing different policy times creates a severe saved-noise confound in early expert layers; corrected by matching step50. L7 MLP contrast remains similar at50/250 (cos0.928), split by successful start grip cos0.430. Candidate: rank-one halve L7 MLP output along time-matched loop-minus-success direction, orthogonalized against successful centroid and initial close-open nuisance direction. No other tensors changed; needs causal and independent evaluation.

## Pass21: L7 MLP rank-one causal probe
Component reduced by0.50004 on first denoising call, but mean XYZ RMS only0.0016 on both16 success/16 loop observations. Two near-zero chunk-mean gripper sign changes. No recovery demonstrated; removed edit. Next inspect direct late visual-to-action heads.

## Next intervention: strengthen late prefix-reading heads
L17 H0/H3/H4/H6 are ~99% action-self; broad attenuation hurt earlier. L17 H1/H7 retain prefix evidence (H1 ~55-77%, H7 ~27-80% on4 diagnostics), unlike the other heads. Double only their o_proj column blocks, preserving self-head denoising. Mechanism hypothesis: late evidence integration underweighted relative to action-state recycling. Requires paired rollout, dose response, and fresh-scene validation.

## Pass22: L17 prefix-head output x2
71 vs70/126 (+0.8+/-3.1pp),8 wins/7 losses; loops48 vs46. L4 -3/18. Rejected/removed.
Next: soften only consistently wrist-focused Q heads by0.5 at L2H7,L8H2/H7,L14H1/H3,L15H2/H7, to change attention collection rather than output amplitude.

## Passes23-24: wrist-query temperature0.5
Confirmed broader spatial attention (e.g. L2H7 drawer-loop effective patches13->221,base mass3%->25%). Fixed-query XYZ RMS0.014-0.086, gripper signs stable. Paired126 success69 vs70 (-0.8+/-3.1pp),7 wins/8 losses; loops44 vs46, slower successes160 vs143 steps. Rejected/removed. Attention concentration alone is not sufficient explanation.

## Next mechanism: irrelevant padded-action denoising feedback
Saved scene979 has EEF height~0.010 and little net displacement while commands remain downward and gripper flips121 times. About57% of loop flips occur inside executed5-action chunks (also58% in successes, so frequency alone is not failure-specific). Input projector cols7:31 represent unexecuted/padded coordinates but contain81.3% of squared weight norm. Candidate halves only these columns, preserving0:6, biases, output heads and all visual pathways. Hypothesis: reduce irrelevant latent contamination of physical action decisions; not established until paired tests.

## Pass25: halve unused action-input columns7:31
72 vs70/126 (+1.6+/-2.8pp),7 wins/5 losses; loops44 vs46 but mean flips16.26 vs14.59 and slower successes156.8 vs142.9 steps. Mechanism unsupported; removed.

## VLM residual sink diagnosis and next causal test
On32 initial observations, VLM residual mean-token energy in channel674 is93.2% atL2,89.1% atL8. LN gamma suppresses it but it still contributes~30-35% of normalized mean energy atL2/L8. Main initial writer is VLM L0 attention: mean channel674=-371 vs output norm407; its o_proj row674 norm17.45 vs row median1.24. Candidate halves only row674 of VLM L0 attention o_proj. Hypothesis: common-mode RMS bottleneck suppresses small visual differences; not established until fixed-input probe and paired recovery/ordinary-proxy tests.

## Passes26-27: halve VLM L0 attention writer row674
Fixed inputs: channel674 is reduced early but partly restored byL8; XYZ RMS0.039-0.310, so this is a causally active pathway. Development126:73 vs70 (+2.4+/-3.6pp), 12 wins/9 losses; loops45 vs46, mean flips13.5 vs14.78; slower success167.7 vs142.9. Not significant. Frozen for first fresh validation batch64 scenes in validation_batch1.json (excluded every previously rolled scene); no tuning based on that batch yet.

## Pass28: independent validation rejected sink edit
124 valid paired episodes:75 vs83 successes (-6.5+/-3.0pp),3 wins/11 losses, loops30 vs22. Removed. Validation batch1 is now inspected; do not reuse as fresh confirmation.
Next analysis: isolate L16 MLP loop component outside dominant normal denoising/initial/success subspace, physical readout span, and sink axes. Twelve matched pairs define geometry; four pairs are reserved for checking feature selectivity before any edit. Intended intervention would route a selective off-normal component into clearance/lift, not scale all actions.

Selective L16 loop-detector analysis failed feature generalization: source loop50 mean1.70 vs normal0.05, but four reserved pairs loop50 mean0.06 vs normal1.79. No edit made from it. Highlights need for independent state/direction checks.

Next causal direction test: visual aperture gain2.0 (opposite of passes17/18 suppression). Same noiseless visual contrast source and offset-preserving rank-one projector formula; all other directions unchanged. Hypothesis: insufficient visual gripper-state precision, not over-reliance; requires fresh validation if development succeeds.

## Pass29: visual aperture gain2.0
Development126:75 vs70 (+4.0+/-3.3pp),11 wins/6 losses; loops44 vs46. L2 +8/86, L3 unchanged, L1 -1/4, L4 -2/18; one pre-fail proxy loss. Best development gain so far but not significant and safety unproven. Frozen for fresh validation_batch2.json; no further tuning yet.

## Pass30: independent aperture-gain validation
128 paired episodes:77 vs77,12 wins/12 losses (0.0+/-3.8pp); loops41 vs38. Development gain did not replicate. Removed aperture edit; no validated repair yet.

## Next: generated-gripper input feedback
Executed grip commands saturate near+/-1 in loops (mean abs0.999, no mass within0.25); 57% of loop flips are inside5-step executions, so simple output threshold bias is unsupported. Arm/grip input and output directions are nearly orthogonal. Candidate halves ONLY action_in_proj column6 (generated/noisy gripper coordinate), preserving observed finger cues, arm coordinates and output head. Test for denoising bistability/chatter.

Pass31 fixed-input generated-gripper half feedback:
initial_success {'xyz_rms': 0.0092, 'grip_sign_change': 0.0063, 'orig_gabs': 0.9994, 'edit_gabs': 1.6695, 'orig_flips5': 0.375, 'edit_flips5': 0.375, 'orig_flips10': 0.625, 'edit_flips10': 0.625}
initial_loop {'xyz_rms': 0.0134, 'grip_sign_change': 0.025, 'orig_gabs': 0.9989, 'edit_gabs': 1.6351, 'orig_flips5': 0.5, 'edit_flips5': 0.4375, 'orig_flips10': 0.625, 'edit_flips10': 0.5625}
later_success {'xyz_rms': 0.0067, 'grip_sign_change': 0.0, 'orig_gabs': 0.9985, 'edit_gabs': 1.5892, 'orig_flips5': 0.125, 'edit_flips5': 0.125, 'orig_flips10': 0.1875, 'edit_flips10': 0.1875}
later_loop {'xyz_rms': 0.0093, 'grip_sign_change': 0.0063, 'orig_gabs': 0.9924, 'edit_gabs': 1.5819, 'orig_flips5': 0.1875, 'edit_flips5': 0.1875, 'orig_flips10': 0.5, 'edit_flips10': 0.5625}

Pass31 rejects generated-gripper half feedback: grip magnitude rises to~1.6, but later-loop sign change only0.6%, flip counts unchanged; XYZ RMS~0.009. Removed without rollout.

## Pass32: contact-specific visual direction
Stable-close observations require preceding10 executed close commands. Within22 scenes, contrast width0.018-0.068m (contact) vs<0.008m (empty closure). 14 source scenes;8 independently reserved for feature check. SigLIP26 wrist mean-unit contrast is centered, LN-scaled, then orthogonalized against earlier aperture direction. Reserved pair cosines all8 positive, mean0.199 (source0.332); base-camera contrast only5/8. Pose changes remain a confound, so call this contact-associated, not a proven object-held detector.
Candidate doubles only this projector input direction with LN-bias offset preservation; orthogonal directions and raw aperture axis unchanged. No success labels used in constructing direction.

Pass33 contact-gain fixed-input action effects:
source_empty {'xyz_rms': 0.0293, 'delta_z': -0.0244, 'orig_close': 0.6, 'edit_close': 0.5, 'sign_changes': 0.1, 'delta_arm_norm': -0.0036}
source_contact {'xyz_rms': 0.0084, 'delta_z': 0.0078, 'orig_close': 0.9286, 'edit_close': 0.9286, 'sign_changes': 0.0, 'delta_arm_norm': 0.0065}
check_empty {'xyz_rms': 0.0159, 'delta_z': -0.0068, 'orig_close': 0.625, 'edit_close': 0.625, 'sign_changes': 0.0, 'delta_arm_norm': -0.0113}
check_contact {'xyz_rms': 0.0065, 'delta_z': 0.0085, 'orig_close': 1.0, 'edit_close': 1.0, 'sign_changes': 0.0, 'delta_arm_norm': 0.0076}

Pass34 contact-projector gain2:72 vs70/126,9 wins/7 losses (+1.6+/-3.2pp), loops46 vs46, L2+6,L3-2,L4-2. Removed; no demonstrated recovery gain.

Next targeted feature-routing hypothesis: stable empty closure vs contact is separable in late VLM features (atL15 all8 reserved paired differences positive, mean6.09 after contact-centroid/aperture/sink projection). This feature also fires on initial opens, so final detector design MUST protect open-wrist, base and language centroids. Probe VLM16 residuals before last-layer V projection. A last-layer VLM v_proj edit affects final expert prefix values but no subsequent VLM cache layers. Intended intervention: selective clearance on empty closures, not global motion scaling.

## Passes35-36: selective empty-closure feature and last-layer gate accounting
VLM16 wrist feature projected outside33-dimensional span of14 source contacts,16 open controls, base/lang centroids and sink674. Reserved empty mean2.284(std1.31), contact0.364(std0.487); 7/8 paired positive. Source empty mean2.915. No success labels used.
Expert17 attention/MLP gates recovered by residual-update accounting on4 inputs; held2 reconstruction error0.29%. Attention gates RMS0.43, mixed signs; ungated routing would be wrong.
Candidate rank-one VLM17 v_proj edit. Gate-corrected weight algebra routes feature through expert17 prefix heads1/7 toward upward action (negative flow Z); zeros first-order readout in all other physical coordinates/heads. Uses nominal clearance0.15, must verify actual direction and safety in paired probe.

Pass37 actual selective-clearance probe results:
source_empty {'dz': 0.0553, 'xy_rms': 0.0058, 'rot_rms': 0.001, 'grip_sign_change': 0.0, 'xyz_rms': 0.0335}
source_contact {'dz': -0.0148, 'xy_rms': 0.0019, 'rot_rms': 0.0003, 'grip_sign_change': 0.0, 'xyz_rms': 0.0094}
check_empty {'dz': 0.0211, 'xy_rms': 0.0029, 'rot_rms': 0.0005, 'grip_sign_change': 0.0, 'xyz_rms': 0.0165}
check_contact {'dz': -0.0099, 'xy_rms': 0.0009, 'rot_rms': 0.0002, 'grip_sign_change': 0.0, 'xyz_rms': 0.0074}
control_open_control {'dz': -0.0146, 'xy_rms': 0.0016, 'rot_rms': 0.0003, 'grip_sign_change': 0.0, 'xyz_rms': 0.0089}
Upstream VLM16 and first-call expert16 unchanged exactly. Feature/dz correlation=0.873. Token-mean protection is not protection of individual attention-weighted patches.

Pass38 clearance0.15 development:71 vs70/126,7 wins/6 losses (+0.8+/-2.9pp), loops43 vs45. No established gain. Next fixed-feature dose0.60 (4x), with probe before rollout.

Pass39 clearance0.60 probe:
source_empty {'dz': 0.2016, 'xy_rms': 0.0226, 'rot_rms': 0.0043, 'grip_sign_change': 0.0, 'xyz_rms': 0.1222}
source_contact {'dz': -0.0621, 'xy_rms': 0.007, 'rot_rms': 0.001, 'grip_sign_change': 0.0, 'xyz_rms': 0.0391}
check_empty {'dz': 0.0786, 'xy_rms': 0.0119, 'rot_rms': 0.0021, 'grip_sign_change': 0.0, 'xyz_rms': 0.062}
check_contact {'dz': -0.0412, 'xy_rms': 0.0036, 'rot_rms': 0.0009, 'grip_sign_change': 0.0, 'xyz_rms': 0.03}
control_open_control {'dz': -0.0593, 'xy_rms': 0.0064, 'rot_rms': 0.0011, 'grip_sign_change': 0.0, 'xyz_rms': 0.0361}
Vertical dose slope=3.539; correlation=0.990.

Clearance0.60 frozen for fresh validation_batch3.json (64 scenes,both grips,seed4006). Excluded all previously rolled scenes and diagnostic inputs. confirmation_batch3.json reserved separately before observing validation results.

## Pass40: fresh clearance0.60 validation neutral
126 valid pairs:77 vs77,9 wins/9 losses (0.0+/-3.4pp); loops40 vs42, mean flips10.22 vs11.87; libero_10 loses4 successes. Not a validated repair.
Known intervention defect from probes: downward leakage in open/contact controls. Next diagnose actual attention-weighted feature via expert17 attn_out difference. Use a normalized channel674 reference to center this feature at the attention readout, not at an unweighted token mean. Centering source is14 source contacts+16 open controls; no success labels or reserved-contact outcomes used. confirmation_batch3 still unseen.

## Passes41-42: correct attention-weighting mismatch
Rank-one attn_out differences decomposed via known o_proj writes; reconstruction error0.115%. Despite zero token-mean scores, actual normal-control H1/H7 scores are~-0.53/-0.41. Reference-only channel674 probe measures per-head common-mode scores. Source-control centering ratios: {1: 1.2369588037994474, 7: -3.718274718743165}. Final candidate is rank2 VLM17 v_proj change: empty feature routing plus head-specific reference subtraction, geometric readout still targets onlyZ throughH1/H7. No success labels in centering. Needs final probe and fresh confirmation; no validated repair yet.

## Pass43: attention-centered clearance0.60 fixed-input probe
source_empty {"n": 14, "dz_mean": 0.4264, "dz_RMS": 0.4779, "dz_scene_p10_p90": [0.1619, 0.6636], "xy_RMS": 0.0533, "rot_RMS": 0.0092, "grip_sign_change": 0.0}
source_contact {"n": 14, "dz_mean": 0.1106, "dz_RMS": 0.173, "dz_scene_p10_p90": [-0.0387, 0.2641], "xy_RMS": 0.0233, "rot_RMS": 0.0039, "grip_sign_change": 0.0}
check_empty {"n": 8, "dz_mean": 0.3075, "dz_RMS": 0.3277, "dz_scene_p10_p90": [0.206, 0.4542], "xy_RMS": 0.0408, "rot_RMS": 0.006, "grip_sign_change": 0.0}
check_contact {"n": 8, "dz_mean": 0.1065, "dz_RMS": 0.1249, "dz_scene_p10_p90": [0.0476, 0.1674], "xy_RMS": 0.0128, "rot_RMS": 0.0047, "grip_sign_change": 0.0}
control_open_control {"n": 16, "dz_mean": 0.1087, "dz_RMS": 0.1418, "dz_scene_p10_p90": [0.0275, 0.2184], "xy_RMS": 0.0185, "rot_RMS": 0.003, "grip_sign_change": 0.0}
Installed edit is rank2 VLM17 v_proj only; relative Frobenius change2.774. Control means alone are not a safety guarantee; inspect RMS and held-out behavior.

## Temporal centering diagnosis after pass43
H7 centered source-contact score grows0.028->2.944 across denoising calls; open-control grows-0.024->2.313. H1 stays selective: empty~3/source and~2.2/check, contact~-0.4, open~0.1. Reconstruction residual grows to10.7% at final call, so later recovered scores are approximate.
New candidate removes H7 routing entirely. Rank1 VLM17 v_proj, H1 route at nominal0.60; normal-source all-call centering coefficient=1.622151. Same14 contact+16 open source controls, no success labels or check outcomes used. Fresh confirmation_batch3 still untouched.

## Pass44: H1-only centered clearance0.60 probe
source_empty {"n": 14, "dz_mean": 0.2318, "dz_RMS": 0.3004, "xy_RMS": 0.0369, "rot_RMS": 0.005, "grip_sign_change": 0.0}
source_contact {"n": 14, "dz_mean": -0.007, "dz_RMS": 0.0487, "xy_RMS": 0.0067, "rot_RMS": 0.001, "grip_sign_change": 0.0}
check_empty {"n": 8, "dz_mean": 0.137, "dz_RMS": 0.1615, "xy_RMS": 0.0231, "rot_RMS": 0.0031, "grip_sign_change": 0.0}
check_contact {"n": 8, "dz_mean": -0.0166, "dz_RMS": 0.0501, "xy_RMS": 0.006, "rot_RMS": 0.0014, "grip_sign_change": 0.0}
control_open_control {"n": 16, "dz_mean": 0.0157, "dz_RMS": 0.046, "xy_RMS": 0.0073, "rot_RMS": 0.0013, "grip_sign_change": 0.0}
H1-only0.60 frozen before outcomes for confirmation_batch3,64 scenes,both grips,seed5006. Fixed-query selectivity improved; this is not yet evidence of recovery success.

## Pass45: H1-only clearance0.60 rejected on untouched confirmation
126 pairs,63 vs69 successes,3 wins/9 losses (-4.8+/-2.7pp). Loop failures40 vs41, total failures63 vs57: reduced loop count is not a recovery gain. Removed. No validated repair.

## Final alternative: empty-grasp reset, not lift alone
Same H1 centered empty-closure feature; direct geometric value route targets nominal opening2.0 and clearance0.15, with all other physical/head first-order constraints zero. No success-label fitting. Needs fixed-input causal check, paired rollout, independent validation. Positive executed gripper is closing (contact controls predominantly positive).

## Pass46: empty-grasp reopening+clearance fixed-input probe
source_empty {"n": 14, "dz": -0.0258, "xy_RMS": 0.058, "rot_RMS": 0.0165, "dgrip": -0.7539, "orig_close_frac": 0.6, "edit_close_frac": 0.3429, "close_to_open": 0.2571, "open_to_close": 0.0}
source_contact {"n": 14, "dz": -0.0043, "xy_RMS": 0.0314, "rot_RMS": 0.0072, "dgrip": 0.0072, "orig_close_frac": 0.9286, "edit_close_frac": 0.9286, "close_to_open": 0.0, "open_to_close": 0.0}
check_empty {"n": 8, "dz": 0.0166, "xy_RMS": 0.0406, "rot_RMS": 0.011, "dgrip": -0.4795, "orig_close_frac": 0.625, "edit_close_frac": 0.475, "close_to_open": 0.15, "open_to_close": 0.0}
check_contact {"n": 8, "dz": -0.0305, "xy_RMS": 0.0258, "rot_RMS": 0.006, "dgrip": -0.0192, "orig_close_frac": 1.0, "edit_close_frac": 1.0, "close_to_open": 0.0, "open_to_close": 0.0}
control_open_control {"n": 16, "dz": 0.0171, "xy_RMS": 0.0189, "rot_RMS": 0.004, "dgrip": -0.0334, "orig_close_frac": 0.05, "edit_close_frac": 0.0375, "close_to_open": 0.0125, "open_to_close": 0.0}

## Pass47: reset full-dose rollout, not yet significant
{"fresh": {"n": 96, "scenes": 48, "original_success": 64, "edited_success": 65, "wins": 9, "losses": 8, "diff_pp": 1.042, "paired_SE_pp": 4.316, "scene_cluster_SE_pp": 4.338, "original_loops": 30, "edited_loops": 29, "t0_solved_pairs": 2}, "prefail_proxy": {"n": 32, "scenes": 16, "original_success": 30, "edited_success": 32, "wins": 2, "losses": 0, "diff_pp": 6.25, "paired_SE_pp": 4.348, "scene_cluster_SE_pp": 4.27, "original_loops": 2, "edited_loops": 0, "t0_solved_pairs": 0}}
Half-dose fixed-input check next; full dose remains frozen for two disjoint fresh batches from frozen_final_pools.json. No candidate selection using these future outcomes.

## Pass48: reset dose-response
source_empty {"half_dgrip": -0.4699, "full_dgrip": -0.7539, "half_close_to_open": 0.1714, "full_close_to_open": 0.2571, "half_contact_or_open_sign_change": 0.1714, "half_xy_RMS": 0.0209}
source_contact {"half_dgrip": 0.003, "full_dgrip": 0.0072, "half_close_to_open": 0.0, "full_close_to_open": 0.0, "half_contact_or_open_sign_change": 0.0, "half_xy_RMS": 0.0122}
check_empty {"half_dgrip": -0.1669, "full_dgrip": -0.4795, "half_close_to_open": 0.05, "full_close_to_open": 0.15, "half_contact_or_open_sign_change": 0.05, "half_xy_RMS": 0.0128}
check_contact {"half_dgrip": -0.0039, "full_dgrip": -0.0192, "half_close_to_open": 0.0, "full_close_to_open": 0.0, "half_contact_or_open_sign_change": 0.0, "half_xy_RMS": 0.0094}
control_open_control {"half_dgrip": -0.0083, "full_dgrip": -0.0334, "half_close_to_open": 0.0, "full_close_to_open": 0.0125, "half_contact_or_open_sign_change": 0.0, "half_xy_RMS": 0.0079}
Full dose restored, frozen for final128+128 paired fresh episodes. Load-error IDs replaced before evaluation from pre-shuffled unused pool: {"451": 600, "455": 923}.

## FINAL DECISION after all50 forward passes
No edits retained. Final reset intervention failed independent validation.
{
  "decision": "REJECT all tested edits; submit original checkpoint",
  "reset_candidate_all_fresh": {
    "n": 352,
    "scenes": 176,
    "original_success": 213,
    "edited_success": 208,
    "wins": 17,
    "losses": 22,
    "diff_pp": -1.42,
    "paired_SE_pp": 1.775,
    "scene_cluster_SE_pp": 1.73,
    "original_loops": 115,
    "edited_loops": 121,
    "t0_solved_pairs": 11
  },
  "reset_candidate_final_confirmation_only": {
    "n": 256,
    "scenes": 128,
    "original_success": 149,
    "edited_success": 143,
    "wins": 8,
    "losses": 14,
    "diff_pp": -2.344,
    "paired_SE_pp": 1.83,
    "scene_cluster_SE_pp": 1.741,
    "original_loops": 85,
    "edited_loops": 92,
    "t0_solved_pairs": 9
  },
  "reset_candidate_excluding_t0_solved": {
    "n": 341,
    "scenes": 174,
    "original_success": 202,
    "edited_success": 197,
    "wins": 17,
    "losses": 22,
    "diff_pp": -1.466,
    "paired_SE_pp": 1.832,
    "scene_cluster_SE_pp": 1.818,
    "original_loops": 115,
    "edited_loops": 121,
    "t0_solved_pairs": 0
  },
  "prefail_proxy_only_not_ordinary_benchmark": {
    "n": 32,
    "scenes": 16,
    "original_success": 30,
    "edited_success": 32,
    "wins": 2,
    "losses": 0,
    "diff_pp": 6.25,
    "paired_SE_pp": 4.348,
    "scene_cluster_SE_pp": 4.27,
    "original_loops": 2,
    "edited_loops": 0,
    "t0_solved_pairs": 0
  },
  "by_level": {
    "L1": {
      "n": 24,
      "scenes": 12,
      "original_success": 15,
      "edited_success": 15,
      "wins": 1,
      "losses": 1,
      "diff_pp": 0.0,
      "paired_SE_pp": 6.019,
      "scene_cluster_SE_pp": 6.155,
      "original_loops": 7,
      "edited_loops": 8,
      "t0_solved_pairs": 8
    },
    "L2": {
      "n": 242,
      "scenes": 121,
      "original_success": 154,
      "edited_success": 150,
      "wins": 13,
      "losses": 17,
      "diff_pp": -1.653,
      "paired_SE_pp": 2.266,
      "scene_cluster_SE_pp": 2.268,
      "original_loops": 77,
      "edited_loops": 79,
      "t0_solved_pairs": 1
    },
    "L3": {
      "n": 48,
      "scenes": 24,
      "original_success": 28,
      "edited_success": 25,
      "wins": 1,
      "losses": 4,
      "diff_pp": -6.25,
      "paired_SE_pp": 4.619,
      "scene_cluster_SE_pp": 3.448,
      "original_loops": 13,
      "edited_loops": 18,
      "t0_solved_pairs": 1
    },
    "L4": {
      "n": 38,
      "scenes": 19,
      "original_success": 16,
      "edited_success": 18,
      "wins": 2,
      "losses": 0,
      "diff_pp": 5.263,
      "paired_SE_pp": 3.671,
      "scene_cluster_SE_pp": 3.617,
      "original_loops": 18,
      "edited_loops": 16,
      "t0_solved_pairs": 1
    }
  },
  "limitation": "A causally controllable visual-to-gripper route was found, but no validated recovery repair.",
  "submitted_edits": [],
  "restoration_verified": true
}
All submission edits removed; load(vname) exactly equals load_original(vname). Candidate archives remain outside edited/ for audit only. No claim of improved recovery or successful localization of a unique loop cause.
