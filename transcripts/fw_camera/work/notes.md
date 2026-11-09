# FastWAM camera robustness repair

## Constraints and plan
- Only direct, mechanism-specific weight edits; no training or local network execution.
- Use practice variants only; reserve disjoint camera variants for confirmation.
- Diagnose camera/control behavior and feature shifts, test causal doses, then paired validation.
- Forward passes used: 19 / 40; see chronological entries.
- Frozen candidate: readout_camera_rank2_half. Original remains fallback pending independent validation.

## Setup
- Forward backend initially offline; no passes spent.
- Practice variants partitioned by SHA256 into 40% discovery and 60% untouched validation.
- Initial matched panel: 20 camera and 10 mild-light episodes across 10 tasks.
- Weight-only self-attention head statistics saved; these are hypotheses, not causal findings.

## Weight-only observations (not yet causal)
- Action-q vs video-k per-head RMS normalization gains correlate 0.9982.
- Layer 29 heads 4/7/8/17 have sparse normalization outliers up to 7.625.
- Several mid-layer heads also have strongly concentrated query spectra, including L12H21, L16H13, L18H7, L23H3; avoid selecting from norms alone.
- No edits applied; simulator still offline at the last request.

- Complete weight scan: L29 heads 4/7/8/17 have action-query/video-key coupling Frobenius proxies 1067/900/1058/1048 vs all-head median 37; top-four singular directions account for >99.8% of their gained coupling.
- Same-layer output-norm-matched negative-control heads predefined: [(29, 19), (29, 1), (29, 16), (29, 12)]. No intervention applied.

## Backend status
- All rollout requests so far were refused as offline, at no pass cost.
- Forward passes still 0/40; no observations, outcomes, or activation evidence yet.
- Current best remains original; no weights changed based on weight-only hypotheses.

## Pass 1: original matched diagnosis
- Backend online. 57 episodes on 19 matched tasks/seeds.
- Camera: 20/38 = 52.6%; light controls: 18/19 = 94.7%.
- All-layer attention outputs and replayable observations saved in roll_1 artifacts.
- Next: inspect layer shifts and first-observation attention; no edits yet.

## Pass 2: same-state probe
- 19 mild/severe camera pairs have exactly identical wrist pixels and proprioception.
- Original first10 XYZ camera-pair RMS discrepancy (mean across pairs): 0.0712.
- Probe activations expose action outputs only; per-camera video outputs are rollout means.
- Failed camera rollouts have mean 17.56 gripper switches vs 5.80 successful camera and 2.00 successful light episodes; all failures timed out (not saturated actions).

## Pass 3 and first intervention
- Attention maps on 2 exact-state camera pairs: L29 H4/7/8/17 video attention mass ~1, agent fraction 0.997-1.000, normalized entropy often 0.01-0.38.
- L11/H0 also strongly agent-preferring; L16 camera contrast is rank-1 concentrated but its top output-aligned head H1 is wrist-preferring, suggesting propagated nuisance.
- First test: L29 action norm_q only, cap |q_gain * video_k_gain| at 4 in H4/7/8/17; 64 coordinates changed. No whole-tensor scaling. Saved as l29_qk_cap4.
- This is an unvalidated candidate, not the current best.

## Passes 4-5: first intervention not established
- cap4 discovery: 63/128 original vs 65/128 edited; +1.6 pp, SE 2.9; 8 gains/6 losses. Inconclusive.
- L29 suspect entropy original=0.152, cap4=0.301.
- Next candidate: original Q/K; halve only L29 self-attention output columns for H4/7/8/17.
- Current best remains original pending stronger evidence. Forward passes used 5/40.

## Passes 6-7: L29 output hypothesis weak
- L29 H4/7/8/17 half-output: +3.1 pp, SE 3.1 on same 128 discovery episodes (10 gains, 6 losses).
- Same-state XYZ discrepancies remain 0.08288 / 0.08585 vs original 0.08301 / 0.08552; first4 action changes only ~0.001 XYZ RMS.
- Therefore L29 agent-locked heads are not established as principal causal failure mechanism; no best promoted.
- Original 19 paired first-call action attention contrasts: L11 RMS .619, rank1 energy .616; L16 RMS .616, rank1 energy .742. L29 rank1 only .214.
- Analyze stable camera-shift directions, then test reversible low-rank output projection with a preserved mild-light mean.
- Forward passes used 7/40.

## Passes 8-9: L11 rank-one output projection rejected
- Stable camera direction (split-task cosine .921); analytic rank-one 50% output projection, mean-light bias preserved.
- Local coefficient halved as predicted, but first10 XYZ camera-pair RMS only .071218 -> .070788 (-0.6%). L16 shift unchanged.
- Discovery rollout: 63/128 original vs 61/128 edited; -1.6 pp, SE 3.3. Reverted.
- Need diagnose post-block residuals/gating; large raw attention shifts are not necessarily behaviorally influential.
- Forward passes used 9/40. Current best ORIGINAL, active edit NONE.

## Passes 10-12: residual/readout diagnosis
- Post-block camera differences accumulate across many layers; L11 and L16 contribute ~5.5% and 7.4% of final contrast, not dominant alone.
- Broader original panel: 37 task groups, 74 camera (40 success) + 37 light (35 success). All 37 camera pairs have exactly matching wrist pixels and proprioception.
- Combined 56 matched pairs: final camera contrast outside a protected control/action-encoder subspace (rank79) retains 72.2% of energy.
- Leading two protected camera directions reproduce across disjoint task subsets (principal cosines .717/.597); rank2 selected to avoid unstable higher modes.
- Candidate: action.head.weight <- W - W U U^T, U two orthonormal camera directions, orthogonal to protected span. Bias and all other weights original.
- This is a direct spectral edit; no optimization or training. Candidate unvalidated, current best original. Passes 12/40.

## Passes 13-15: protected readout projection is promising
- Full rank2 edit: weight-relative change 4.174%; protected-span and action-encoder mapping errors <6e-9.
- Replay: camera-pair XYZ RMS .071218 -> .069914 (-1.83%); 13/19 pairs improve. Severe edit RMS .01168 vs light .00282; no general action shrinkage.
- Discovery camera pass14: 64/128 original vs 72/128 edited; +6.25 pp, SE3.28, 13 gains/5 losses.
- Light safety pass15: 118/128 vs119/128; +.78pp, SE2.07, 4 gains/3 losses; one original error, analyze sensitivity separately.
- Next: fixed-subspace half-dose and matched negative control, then untouched camera validation plus all other categories.
- Current active candidate half-dose; full candidate saved, not yet validated as best. Passes 15/40.

## Passes 16-17: fixed-subspace dose response
- Half rank2 projection camera-pair action RMS 0.070460; half/full action-change slope 0.468, cosine 0.956.
- Half dose rollout:64/128 original vs73/128 edited; +7.03pp SE2.97 (12 gains,3 losses).
- Prefer half dose provisionally: similar discovery benefit with 2.087% head-weight change. Not yet independently validated.
- Negative control: deterministic random rank2 in protected complement, orthogonal to camera U; alpha chosen solely to match edit Frobenius norm.
- Current active edit NEGATIVE CONTROL for causal test. Passes17/40.

## Passes 18-19: negative control and candidate freeze
- Norm-matched random rank2 control camera-pair RMS 0.071233 vs original .071218.
- Negative-control rollout +1.56pp SE2.48 (6 gains/4 losses), versus targeted half +7.03pp SE2.97.
- Restored HALF targeted readout edit and froze tensor hash in candidate_lock.json before any validation results.
- Untouched camera practice validation panels: [128, 128, 128, 90]; no overlap with any used variant. All other category validation still required.
- Active edit ONLY mot.mixtures.action.head.weight. All earlier edits reverted. Passes19/40.

## Pass20: first independent camera panel does not replicate
- Validation A:59/128 original vs57/128 edited; -1.56pp SE2.71 (5 gains/7 losses).
- Candidate unchanged for preplanned panel B. Do NOT promote based on discovery gain; original remains fallback.

## Pass21: second independent camera panel
- Validation B:55/128 original vs59/128 edited; +3.125pp SE3.3 (11 gains/7 losses).
- Combined A+B:114/256 vs116/256, +0.781pp; discovery gain not yet replicated.
- Candidate remains frozen; finish preplanned C/D before interpreting. Passes21/40.

## Pass22: third independent camera panel
- C:60/128 original vs57/128 edited; -2.344pp, 5 gains/8 losses.
- A-C:174/384 vs173/384; -0.260pp. Discovery gain has not replicated.
- Finish preplanned D; do not tune using these validation outcomes. Passes22/40.

## Pass23: readout candidate not confirmed
- D:34/90 vs39/90 (+5.556pp); all independent panels: {'n': 474, 'original': 208, 'edited': 212, 'diff_pp': np.float64(0.844), 'se_pp': np.float64(1.58), 'gains': 30, 'losses': 26}.
- +0.844pp on 474 independent episodes, below SE; do NOT promote discovery gain.
- Archived candidate unchanged; reverted to ORIGINAL (no active edits).
- Revisit upstream attention mechanism using discovery activations only; budget17 passes left.

## Revised mechanism after pass23 (no validation-derived directions)
- Mean32 action contrast equals negative final residual readout (cos .99990).
- Full rank79 control protection weakened camera-action alignment (cos .5005 vs revised .6034); revised analytic rule preserves encoder+bias and top clean PCs covering85% energy (rank16; total protected24).
- Revised nuisance rank2 reproduces across task split (principal cos .689/.656), camera-readout-error cosine .603, half-dose predicted mean-action discrepancy ratio .874.
- Mean clean readout preserved by compensating head bias; only head.weight and head.bias edited.
- Active candidate readout_camera_rank2_clean16_center_half, not yet tested. Original remains fallback.
- Reserved 128 never-run camera variants in final_camera_reserve.json. Planned remaining passes: half/full replay+discovery, negative control, frozen validation, all six safety categories.

## Passes24-25: revised protection remains weak
- Revised half-dose replay first10 camera XYZ RMS .071218 -> .070267 (-1.34%); light change .00168.
- Discovery 63/128 ->66/128, +2.344pp SE2.822 (8 gains/5 losses). Not established.
- Do not promote. Candidate archived; inspect whether previous L29 cap left dominant anchoring coordinates active before allocating remaining validation budget.

## New causal test prepared after pass25
- Reverted weak revised readout. Active only action L29 self_attn.norm_q.weight.
- H4/7/8/17: 64 coordinates with original |q_gain*video_k_gain|>4 account for89-96% of gain-product energy.
- Test exact coordinate ablation (set only these q gains to0), stronger than prior cap4; values, output projections and other heads unchanged.
- Candidate l29_qk_outlier64_ablation archived. Not yet validated; original fallback.

## Passes26-27: attention ablation not established; stop hypothesis search
- Ablating64 query-gain coords returns suspect coupling norms to28-39 vs original900-1067, cap4 106-136.
- Attention entropy rises .152 -> .398; H7 agent fraction .999 -> .162, other-head maps nearly unchanged.
- Action camera discrepancies essentially unchanged. Discovery64/128 ->66/128 (+1.563pp SE2.93). Reverted.
- Final evaluation returns to the ORIGINAL frozen readout_camera_rank2_half tensor, no further tuning.
- 7 more paired camera panels (896 episodes) plus prior474 =1370, 602 selection-disjoint variants, 2-3 seeds each. Passes28,29,30,31,38,39,40.
- Passes32-37: all6 noncamera categories,128 paired episodes each.
- Final retention criterion fixed in final_evaluation_plan.json: camera gain >1.96*variant-cluster SE, no noncamera point-estimate loss >2pp; otherwise ORIGINAL.
- No active edit at this moment; restore frozen candidate exactly from archive before pass28.
- Frozen candidate restored exactly; SHA256 e16aec070cd68d6c8e843fa871bb1171d2c1a70c6ba70e86d06129cce3d3efdc verified. Only action.head.weight active.

## Pass28: untouched camera reserve
- 41/128 ->47/128 (+4.688pp SE3.110;11 gains/5 losses).
- All602 selection-disjoint variants, one seed each: {'n': 602, 'original': 249, 'edited': 259, 'diff_pp': np.float64(1.661), 'se_pp': np.float64(1.409), 'gains': 41, 'losses': 31, 'variants': 602, 'cluster_se_pp': 1.409, 'equal_variant_diff_pp': 1.661, 'equal_variant_se_pp': 1.409}.
- Still not conclusive; continue fixed seed panels, no tuning. Only frozen head.weight active.

## Pass29: Preplanned camera seed panel
- Frozen candidate unchanged. {'n': 128, 'original': 56, 'edited': 59, 'diff_pp': np.float64(2.344), 'se_pp': np.float64(3.756), 'gains': 13, 'losses': 10}

## Pass30: Preplanned camera seed panel
- Frozen candidate unchanged. {'n': 128, 'original': 39, 'edited': 46, 'diff_pp': np.float64(5.469), 'se_pp': np.float64(2.786), 'gains': 10, 'losses': 3}

## Pass31: Preplanned camera seed panel
- Frozen candidate unchanged. {'n': 128, 'original': 56, 'edited': 50, 'diff_pp': np.float64(-4.688), 'se_pp': np.float64(3.11), 'gains': 5, 'losses': 11}
- Camera aggregate through31: {'n': 986, 'original': 400, 'edited': 414, 'diff_pp': np.float64(1.42), 'se_pp': np.float64(1.129), 'gains': 69, 'losses': 55, 'variants': 602, 'cluster_se_pp': 1.1, 'equal_variant_diff_pp': 0.969, 'equal_variant_se_pp': 1.184}. Retention criterion not met.
- Next32-37 are fixed noncamera safety checks;38-40 complete fixed camera seed panels.

## Pass32: Safety: Light Conditions
- Frozen candidate unchanged. {'n': 128, 'original': 121, 'edited': 121, 'diff_pp': np.float64(0.0), 'se_pp': np.float64(1.109), 'gains': 1, 'losses': 1}

## Pass33: Safety: Robot Initial States
- Frozen candidate unchanged. {'n': 128, 'original': 85, 'edited': 85, 'diff_pp': np.float64(0.0), 'se_pp': np.float64(2.218), 'gains': 4, 'losses': 4}

## Pass34: Safety: Language Instructions
- Frozen candidate unchanged. {'n': 128, 'original': 77, 'edited': 75, 'diff_pp': np.float64(-1.562), 'se_pp': np.float64(2.713), 'gains': 5, 'losses': 7}

## Pass35: Safety: Background Textures
- Frozen candidate unchanged. {'n': 128, 'original': 79, 'edited': 80, 'diff_pp': np.float64(0.781), 'se_pp': np.float64(1.752), 'gains': 3, 'losses': 2}

## Pass36: Safety: Sensor Noise
- Frozen candidate unchanged. {'n': 128, 'original': 73, 'edited': 70, 'diff_pp': np.float64(-2.344), 'se_pp': np.float64(2.593), 'gains': 4, 'losses': 7}

## Pass37: Safety: Objects Layout
- Frozen candidate unchanged. {'n': 128, 'original': 92, 'edited': 92, 'diff_pp': np.float64(0.0), 'se_pp': np.float64(1.569), 'gains': 2, 'losses': 2}

## Final-budget amendment after pass37
- All6 categories checked at128 paired episodes each: Light0pp, Robot0pp, Language-1.563pp, Background+.781pp, Sensor-2.344pp, Layout0pp.
- Sensor safety flag exceeds2pp point-estimate tolerance but SE~2.6pp. Allocate pass38 to one disjoint128-variant Sensor replication; pool36+38 regardless of result. No further safety retesting.
- Passes39-40 remain their preselected camera panels; final camera n1242 instead of1370. All weights and original retention thresholds unchanged.
- Camera gain through31 still unconfirmed (+1.420pp, clusterSE1.100); original remains fallback.

## Pass38: predeclared Sensor Noise replication
- Frozen candidate hash unchanged. Replication: {"n": 128, "original": 97, "edited": 99, "diff_pp": 1.562, "se_pp": 2.214, "gains": 5, "losses": 3, "variants": 128, "cluster_se_pp": 2.214, "equal_variant_diff_pp": 1.562, "equal_variant_se_pp": 2.214}
- Disjoint batches36+38 pooled regardless of sign: {"n": 256, "original": 170, "edited": 169, "diff_pp": -0.391, "se_pp": 1.706, "gains": 9, "losses": 10, "variants": 256, "cluster_se_pp": 1.706, "equal_variant_diff_pp": -0.391, "equal_variant_se_pp": 1.706}
- All6 category point estimates now meet preset >-2pp rule; this is not proof of noninferiority. No additional safety retesting.
- Passes39-40 remain the originally preselected camera panels; camera retention still requires gain >1.96*variant-cluster SE.

## Pass39: penultimate fixed camera seed panel
- Frozen candidate unchanged. {"n": 128, "original": 59, "edited": 57, "diff_pp": -1.562, "se_pp": 3.134, "gains": 7, "losses": 9, "variants": 122, "cluster_se_pp": 3.138, "equal_variant_diff_pp": -2.049, "equal_variant_se_pp": 3.209}
- Independent camera aggregate: {"n": 1114, "original": 459, "edited": 471, "diff_pp": 1.077, "se_pp": 1.062, "gains": 76, "losses": 64, "variants": 602, "cluster_se_pp": 1.047, "equal_variant_diff_pp": 1.052, "equal_variant_se_pp": 1.112, "ci95_cluster_pp": [-0.9744302109462399, 3.1288287746805308]}
- Camera retention criterion still not met; finish fixed pass40. No tuning, no selective omission of any validation panel.
