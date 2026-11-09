# FastWAM language-robustness weight-edit investigation

## Constraints and plan
- No training, backward passes, or policy execution in the Python kernel.
- Only practice variants and forward-tool outputs are inspected.
- Diagnose language failures against less perturbed variants of the same tasks.
- Test localized edits, include causal dose-response and untouched validation.
- Require paired language gains larger than paired uncertainty and check all other categories.
- Forward budget: 40 passes. No passes used yet.

## Current submitted edit
None.


## Backend status
Initial rollout refused while backend offline; no pass charged. Inspecting text-conditioning weights before retrying.

## Weight-only screening (not causal evidence)
- Cross-attention L13 H14 has largest within-layer V/O Frobenius gain: 2.81x median in action, 2.59x in video. L15 H4 next (1.94x/1.81x).
- Final self-attention norm outlier channel 606 is shared by both experts; not assumed to be a language defect.
- Practice variants assigned deterministically to discovery (40%), validation (30%), confirmation (30%); no outcomes seen yet. Initial random baseline, if run, is diagnostic only.

- Exact head OV spectra reveal many video-specific late rank-one modes (e.g. L23 H3: s1=81.92, 97.7% energy; L24 H21: 64.77, 95.9%; L13 H7: 44.98, 76.5%). These are not evidence of a fault without activations.
- No weights edited. Backend still offline after three free retries.

- Late video OV outlier modes largely share a coordinate: common PC has 0.99582 loading at residual channel 1938, |cos(ones)|=0.01064. This is a candidate massive-activation channel, not a proven defect. Weight-only next-layer reader screening highlights action L24 H17 (1.65x random-direction value sensitivity), L16 H14 and L14 H3. Must isolate language-sensitive routes rather than ablate the channel.

## Pass plan
Aim to spend <=16 passes on diagnosis and causal discovery, then lock the edit for remaining paired language and six-category validation. Exclude all discovery variants from novel-variant language estimates. If evidence is weak, retain the original rather than an unjustified edit.

## Candidate mechanisms to discriminate with fixed-input probes
1. Direct action text-attention distraction / over-sharp attention.
2. Paraphrase-sensitive video text writes, possibly residual channel 1938, then shared-attention reads into actions.
3. Earlier text-embedding style sensitivity. Text matrices have no overwhelming single singular direction in static analysis.
No mechanism is accepted until paired same-observation probes and rollouts.

## Backend wait
Still offline after 10 weight-analysis code runs. No charged forward passes and no edits. Static analysis is complete enough to define causal tests; pausing briefly rather than drawing conclusions without behavior.

- Prepared unsubmitted rank-one head-output intervention factors in rank_one_intervention_templates.pt. O_head' = O_head - alpha*u*(u^T O_head), where u is an effective OV left singular mode. This is closed-form weight analysis, not fitting. Factors include second-mode controls. No intervention will be selected without behavior. Backend still offline.

- Algebraic template safety check: half-strength L23 H3 rank-one projection halves its leading effective OV singular value while preserving the remaining spectrum to float32 precision. This is not a behavioral test. No active edits. Simulator still unavailable; waiting for startup.

## Additional static negative results
- Action weights are not simple top-left crops of video weights (tested text embeddings and attention projections; cosine near zero).
- Removing video output coordinate 1938 algebraically reduces most late exceptional head OV singular values to ordinary ranges. This confirms the source of the spectral outliers, not their harmfulness.
- No behavioral evidence is available yet; forward tool remains offline. All 40 passes remain. No edited weights are active.

## Pass 1 baseline (backend available)
16 original language episodes: 12 successes, 4 failures. Failures: spatial1062 (noun circumlocutions), goal1188 (wine container/cupboard), goal1246 (implicit opening prerequisite), long1208 (coffee conversational phrasing). Descriptive only. Replay inputs reside in per-episode symlink directories. No edits. 1/40 passes used.

## Pass 2 controls
13 same-task canonical-instruction lighting controls: 12/13 success. All four language-failure tasks succeed in controls. Lighting differs, so contrast_inputs contains 16 exact-observation pairs: original language input vs prompt copied from same-task practice control, identical pixels/proprio. Will probe with fixed diffusion noise. 2/40 passes used.

## Probe restriction and pass 3
Constructed instruction-pair probe refused without charge: tool requires unmodified rollout inputs. Deleted constructed copies; no attempt to bypass. Use natural language-variant pairs with identical initial image/proprio.
Pass 3 original language, n=128 seed2713: 90/128=70.31%; long21/38, goal15/29, object28/28, spatial26/33; diff5=10/24. These and pass1 are discovery only. 3/40 used, no edits.

## Pass 4 controlled natural-pair probe
20 unmodified first-observation inputs: 10 same-task success/failure instruction pairs. All paired images and proprioception checked exactly equal; fixed diffusion noise seed7. Recorded all 30 residual layers. Includes 2 long-horizon task pairs. 4/40 used, no edits.

## Passes 5-6 diagnosis
- Pass 5 attention maps expose action-to-video only; two identical-observation instruction pairs show large visual-attention redistribution at L10-15 (individual head L1 changes up to 0.96). Not causal yet.
- Pass 4 residual contrasts grow sharply at L11 (increment RMS 0.0723 vs L10 0.0227); relative residual difference reaches ~10% late. L13 H14 static OV gain is broad-spectrum, not a rank-one sink. Its top six action OV modes explain only 0.53% of L13 residual contrast increment, so a rank-one edit there is not supported.
- Pass 6 records original attention outputs for all ten natural instruction pairs, all layers. No active edits. 6/40 passes used.

## First causal candidate
- Candidate L13H14_both_half: retain 50% of cross-attention O columns [1792:1920] in video and action L13, all other weights unchanged. This is head-local, not whole-tensor scaling.
- Hypothesis: unusually broad-spectrum high OV gain amplifies instruction wording differences. Static head gain alone does not establish harm; paired discovery rollout is the causal test.
- Pass 6 attn-output contrast peaks at L11 (RMS .503), with substantial downstream differences L15-17 and L27. Failed pair trajectories often continue moving with prolonged one-sided gripper states; no claim of semantic localization from these alone.

## Pass 7 causal negative result
- L13 H14 output gain 0.5 in both experts, paired same 128 discovery variants: original 90/128, edited 89/128; -0.8 pp +/-2.6 pp SE; 5 rescues, 6 regressions. Head gain hypothesis not supported. REVERTED all edits.
- Next localization: paired instruction-contrast directions at L11 attention outputs; seek a low-rank, task-preserving intervention and test its action prediction effect before rollout.

## Pass 8 and protected projection candidate
- Original attention-output probes on 48 registered saved inputs; 37 successful references cover 36 tasks. Refused convenience directory deleted; actual probe used only original file paths.
- Closed-form candidate: remove successful-output span from the ten paired instruction differences, normalize pair weights, take leading PCA direction u at L11. Modify action L11 self_attn.o weight and bias by (I - 0.5 u u^T). No optimizer, loss, gradients, or policy execution in kernel. This protects reference mean outputs algebraically, not all tokens or future states; must test behavior.

## Pass 9 causal localization, negative behavioral result
- Protected action L11 self-attention rank-one half-projection attenuated the measured component to 0.5004 of original, confirming attn_out records the self-attention output.
- Mean action RMS shifts only .00061 for both successful and failed prompts; paired action gap .06923 -> .06931 (no improvement). REVERTED. Removing error directions orthogonal to the task span has too little behavioral leverage here.
- Next predefined branch-discrimination tests: L13 H14 video-only and action-only output attenuation. Combined pass-7 edit may mask opposing effects; no success claimed.

## Pass 10
- Video-only L13 H14 half-gain: original 89/128, edited 89/128; 0.0 pp +/-1.6 SE; 2 rescues, 2 regressions. No benefit. Reverted before action-only test.
- Original repeated baseline differs slightly from pass 3/7 despite same requested sample; use within-pass paired estimates rather than across-pass outcome subtraction.

## Pass 11
- Action-only L13 H14 half-gain: original 89/128, edited 89/128; 0.0 pp +/-1.6 SE; 2 rescues, 2 regressions. Reverted. Combined and branch-isolated tests provide no support for this candidate.
- Investigating video-channel-1938 text-write/shared-attention-read route. Episode-average contrast is confounded by different trajectories; requires fixed-input/causal confirmation. 11/40 passes used, no accepted edit.

## Video text-write / action-read candidate
- Episode-averaged same-task video contrasts have channel-1938 energy fractions L13 .384, L20 .498, L22 .724, L23 .472. Correlation is trajectory-confounded. In 128 discovery episodes L23 agent-channel mean is 59.29 (success) vs 52.11 (failure); prompt-length correlation -.241.
- Static route: video L23 cross-attn H3 OV leading singular value 81.91 (97.65% energy), left mode |u1938|=.9964. Next layer strongest value read is H17, 1.65x random-direction sensitivity.
- Candidate is source-local rank-one half attenuation of video L23 cross_attn.o H3, NOT global channel ablation. Preserves other effective OV modes. Awaiting causal rollout.

## Pass 12 causal negative result
- Video L23 H3 leading OV mode half-strength: original 90/128, edited 88/128; -1.6 pp +/-2.2 SE; 3 rescues, 5 regressions. Reverted; suppressing this source does not improve robustness.
- Considering opposite source dose because failed prompts had smaller common-mode signal; no conclusion from correlation alone.

## Action visual-grounding candidate from fixed-input attention maps
- At L11 (first sharp residual divergence), H19 has greatest mean visual distribution L1 difference across two fixed-observation pairs: .678. Both failed prompts concentrate attention: peak mass .448->.609 and .445->.606; entropy 1.802->1.195 and 1.755->1.535 nats.
- Candidate: action L11 self_attn.norm_q.weight indices [2432:2560] scaled to .75. Only 128 query-scale entries change; no whole tensor scaling, no value/output deletion. Predict higher H19 attention entropy and reduced prompt-induced visual misdirection. Await fixed-input causal probe.

## Passes 13-14: head-local visual-attention temperature
- Pass 13 fixed-input causal probe: action L11 H19 query scale .75 raises entropy in all four inputs (e.g. failed prompts 1.195->1.660, 1.535->2.284), decreases peaks .609->.504, .606->.446. Visual-pair L1 .489->.513 and .867->.662. Initial actions shift only ~.0006 RMS; no consistent action-gap improvement.
- Pass 14 paired discovery rollout: 90/128 original vs 91/128 edited; +0.8 pp +/-1.4 SE, 2 rescues/1 regression. Inconclusive, not an accepted edit. Next test is stronger .50 query scale as a predefined dose-response.

## Pass 15 and geometric follow-up
- H19 query scale .50: 90/128 original vs 86/128 edited; -3.1 pp +/-1.9 SE (1 rescue, 5 regressions, including one error). Reverted. Excessive softening harms spatial precision; .75 result not accepted.
- Follow-up to pass 9: orthogonal deflation removes only the task-orthogonal component of a wording contrast, leaving its task-overlapping component. Construct rank-one oblique projection from the same normalized paired differences: u = protected contrast PC, d = corresponding full contrast with u^T d=1. Edit action L11 self-attention O and bias by (I-alpha*d*u^T), alpha=.5. Successful reference mean outputs are algebraically preserved. Closed-form PCA/projector, no regression targets, loss fitting, or optimization. Awaiting causal probe.

## Pass 16
- Tested oblique protected L11 projector at alpha .5 on all ten original instruction pairs; results saved in probe16_comparison.json. Reverted pending independent assessment; do not treat suppression of a diagnostic activation as proof of semantic repair.
- Discovery limit reached: next stage will validate the minimal L11 H19 query-scale .75 candidate on variants excluded from all original diagnostic rollouts. It is only provisional (+.8 pp discovery, below SE), and will be rejected unless independent gain is clear and other categories are safe.

## Locked provisional candidate / validation plan
- Pass 16 oblique projector: mean action gap .06923->.06933, successful/failed shifts .00064/.00063, no benefit. Reverted.
- Active provisional edit: ONLY action L11 self_attn.norm_q.weight [2432:2560] at .75 original. Backup candidate_L11H19_query075.pt. No claim of improvement yet.
- Frozen novel-variant language plan in locked_language_validation_plan.json: five disjoint batches (128,128,128,128,119), 631 variants, all excluded from diagnostic variants. First three validation, final two confirmation. Candidate will not be tuned on these outcomes.
- Retention criterion: independent paired gain clearly > uncertainty (target >2 SE), plus no meaningful non-language harm. Otherwise submit original. 16/40 forward passes, 51 code runs used after this block.

## Pass 17: frozen language validation batch 1
- Verified only L11 H19 query-scale .75 remains active (128 entries).
- Previously unused variants: original 76/128, edited 73/128; -2.3 pp +/-2.3 pp SE, 3 rescues/6 regressions. No benefit established.
- Candidate unchanged; continue frozen validation, do not tune on these outcomes.

## Pass 18: frozen language validation batch 2
- Original 77/128, edited 79/128; +1.6 pp +/-1.9 pp SE, 4 rescues/2 regressions.
- Combined batches 1-2: {"n": 256, "original_successes": 153, "edited_successes": 152, "diff_pp": -0.3906, "se_pp": 1.5157, "rescues": 7, "regressions": 8}
- Run the third predeclared validation batch before deciding whether confirmation is warranted; leave the last 247 variants untouched for now.

## Pass 19: query-softening candidate REJECTED
- Three predeclared validation batches completed: {"n": 384, "original_successes": 240, "edited_successes": 236, "diff_pp": -1.0417, "se_pp": 1.1649, "rescues": 8, "regressions": 12}
- All edits reverted. No demonstrated language improvement.
- Confirmation batches 4-5 (247 variants) remain entirely unrun. Do not tune on validation episodes or inspect their detailed failure patterns.
- Remaining 21 passes: allow at most five renewed discovery/causal passes using the original discovery data; reserve two for untouched language confirmation, twelve for six-category checks, and two for seed replication. Any new candidate must pass independent confirmation; otherwise submit original.

## Renewed causal test: source-mode amplification
- Existing fixed-observation residual probes expose action layers only, so they do not verify a video channel-1938 deficit.
- Test the opposite sign of pass 12's source-local edit: leading effective OV mode of video L23 cross-attention H3 at gain 1.5. Formula O_head' = O_head + .5*u*(u^T O_head); all other modes preserved.
- Motivated by the earlier discovery-only signal deficit correlation, not by rejected-candidate validation failures. This is a hypothesis test.
- Predeclare gain 2.0 as the stronger dose if warranted; only original discovery variants may be used for choosing a candidate.

## Pass 20: source-mode gain 1.5, no net improvement
- Original discovery variants: {"n": 128, "original_successes": 90, "edited_successes": 90, "diff_pp": 0.0, "se_pp": 2.4802, "rescues": 5, "regressions": 5}
- Episode-mean channel-1938 changes: {"22": {"n": 128, "original_ch1938": 63.59206336736679, "edited_ch1938": 63.65883031487465, "mean_residual_shift_rms": 0.016790282134934387}, "23": {"n": 128, "original_ch1938": 57.15802678465843, "edited_ch1938": 56.453788965940475, "mean_residual_shift_rms": 0.022369916183379246}, "24": {"n": 128, "original_ch1938": 43.19974163174629, "edited_ch1938": 43.161019027233124, "mean_residual_shift_rms": 0.015026395249151392}}; trajectory-confounded, not fixed-input proof.
- Apply the predeclared 2.0x leading-mode dose for the next causal test. No edits to other heads or modes.

## Pass 21: video-source hypothesis not supported
- Gain 2.0 result: {"n": 128, "original_successes": 91, "edited_successes": 89, "diff_pp": -1.5625, "se_pp": 2.7134, "rescues": 5, "regressions": 7}
- Gains .5 (pass 12), 1.5 (pass 20), and 2.0 (pass 21) show no reliable benefit. All source edits reverted.
- At most three renewed exploratory passes remain before final candidate lock. No successful repair yet.

## Final exploratory candidate: direct text-write mode at action L11 H5
- On the original ten fixed-observation instruction pairs, head O energy alignment with the L11 residual contrast increment is 12.56x isotropic (next head 2.17x). Its leading O left singular vector explains 59.35% of contrast energy, 19.27% with equal pair weighting; effect is concentrated in a few pairs.
- This is the leading O mode, NOT the leading effective OV mode (the latter explains only 7.59% energy).
- Candidate: O_head' = O_head - .5*u*(u^T O_head), action L11 cross-attention H5 only; no bias edit. All orthogonal output modes preserved.
- Pass 22: fixed-input causal probe, predicting unchanged L10 and reduced L11 contrast along u. Pass 23: half-retention discovery rollout. Pass 24: zero-retention stronger dose, then end exploration. No candidate will be retained on discovery results alone.

## Pass 22: direct text-mode fixed-input causal probe
- {"layer10_first_step_mean_rms_shift": 0.0, "layer10_all_steps_mean_rms_shift": 0.0008340083921211771, "L11_mode_contrast_ratio_first_step": 0.5043938111400462, "L11_mode_contrast_ratio_step_mean": 0.5065681093031994, "mean_action_pair_gap_original": 0.06923426687717438, "mean_action_pair_gap_edited": 0.06869190186262131, "successful_prompt_action_shift_mean": 0.0010188695741817355, "failed_prompt_action_shift_mean": 0.0013795748818665743, "pairs_with_smaller_action_gap": 2, "pair_gap_original": [0.23127469420433044, 0.16070081293582916, 0.07713480293750763, 0.04779098927974701, 0.04012506082653999, 0.03578652814030647, 0.030839288607239723, 0.025689784437417984, 0.021951641887426376, 0.02104910835623741], "pair_gap_edited": [0.23127591609954834, 0.16100122034549713, 0.07721976935863495, 0.04793308302760124, 0.03333057090640068, 0.035839371383190155, 0.0309719517827034, 0.026105573400855064, 0.02219950594007969, 0.02104213461279869]}
- Activation attenuation alone is not proof of a useful repair. Proceed to the predeclared paired discovery rollout.

## Pass 23: direct text-mode half-retention discovery rollout
- {"n": 128, "original_successes": 88, "edited_successes": 91, "diff_pp": 2.3438, "se_pp": 2.593, "rescues": 7, "regressions": 4}
- Positive estimate is smaller than its SE; not accepted. Original within-pass baseline 88/128 differs from earlier repeats, so do not compare unpaired totals across passes.
- Saved half-retention candidate; apply zero retention of the same single mode for the final predeclared exploratory dose test.

## Pass 24 and FINAL CANDIDATE LOCK
- Full removal of action L11 H5 leading O mode: {"n": 128, "original_successes": 90, "edited_successes": 86, "diff_pp": -3.125, "se_pp": 2.7028, "rescues": 4, "regressions": 8}
- Restore half retention for evaluation, not acceptance. Fixed-input mode contrast was reduced to .5044 (first step), but discovery success gain +2.3 pp was below SE.
- No further weight tuning. Remaining 16 passes: two batches of untouched language variants (128+119) at seed7, twelve safety batches (256 unique variants per other category), and the same untouched language variants at seed19. Final decision uses all predeclared results.
- If independent gain is not clearly larger than paired uncertainty or there is material non-language damage, submit original.

## Pass 25: first untouched direct-text-mode test
- {"n": 128, "original_successes": 88, "edited_successes": 88, "diff_pp": 0.0, "se_pp": 2.2184, "rescues": 4, "regressions": 4}
- No net benefit. Candidate remains frozen pending the full evaluation.
- Non-language safety plan frozen in locked_safety_evaluation_plan.json: 256 previously unseen variants per category, deterministic hash selection, two disjoint batches each, seed7. No safety outcomes used in selection.

## Pass 26: untouched language seed7 completed
- {"n": 247, "original_successes": 164, "edited_successes": 162, "diff_pp": -0.8097, "se_pp": 1.7204, "rescues": 8, "regressions": 10}
- No independent language benefit. Do not tune the edit. Complete the predeclared safety and seed-replication tests, then apply retention rule.
- Live candidate is ONLY action L11 H5 cross-attention leading O mode at half retention. Full evaluation ledger: locked_evaluation_results.json.
- Safety pass mapping: 27-28 Light, 29-30 Robot Initial States, 31-32 Sensor Noise, 33-34 Camera, 35-36 Background, 37-38 Objects Layout. Passes 39-40 repeat language batches at seed19.

## Pass 27: safety Light Conditions, batch 1
- {"n": 128, "original_successes": 120, "edited_successes": 122, "diff_pp": 1.5625, "se_pp": 1.1005, "rescues": 2, "regressions": 0}
- Candidate unchanged. Full cumulative results in locked_evaluation_results.json.

## Pass 28: safety Light Conditions, batch 2
- {"n": 128, "original_successes": 117, "edited_successes": 119, "diff_pp": 1.5625, "se_pp": 2.2141, "rescues": 5, "regressions": 3}
- Candidate unchanged. Full cumulative results in locked_evaluation_results.json.

## Pass 29: safety Robot Initial States, batch 1
- {"n": 128, "original_successes": 88, "edited_successes": 83, "diff_pp": -3.9062, "se_pp": 2.5781, "rescues": 3, "regressions": 8}
- Candidate unchanged. Full cumulative results in locked_evaluation_results.json.

## Pass 30: safety Robot Initial States, batch 2
- {"n": 128, "original_successes": 96, "edited_successes": 93, "diff_pp": -2.3438, "se_pp": 2.593, "rescues": 4, "regressions": 7}
- Candidate unchanged. Full cumulative results in locked_evaluation_results.json.

## Interim retention assessment after pass 30
- Language seed7: -0.81 pp +/-1.72 SE, no independent gain.
- Robot Initial States: observed -3.125 pp over 256 paired variants, above the allowed harm scale. Candidate currently fails retention criteria.
- Continue the frozen evaluation solely for complete evidence. Do not compensate with further weight edits. Original remains the default final choice.

## Pass 31: safety Sensor Noise, batch 1
- {"n": 128, "original_successes": 82, "edited_successes": 80, "diff_pp": -1.5625, "se_pp": 2.2141, "rescues": 3, "regressions": 5}
- Restart verified the exact frozen candidate tensor hash. No tuning; continue the locked evaluation.

## Pass 32: safety Sensor Noise, batch 2
- Batch: {"n": 128, "original_successes": 80, "edited_successes": 75, "diff_pp": -3.9062, "se_pp": 2.3273, "rescues": 2, "regressions": 7}
- Category cumulative: {"n": 256, "original_successes": 162, "edited_successes": 155, "diff_pp": -2.7344, "se_pp": 1.6046, "rescues": 5, "regressions": 12}
- Candidate unchanged; no outcome-dependent tuning.

## Pass 33: safety Camera Viewpoints, batch 1
- Batch: {"n": 128, "original_successes": 42, "edited_successes": 42, "diff_pp": 0.0, "se_pp": 2.717, "rescues": 6, "regressions": 6}
- Category cumulative: {"n": 128, "original_successes": 42, "edited_successes": 42, "diff_pp": 0.0, "se_pp": 2.717, "rescues": 6, "regressions": 6}
- Candidate unchanged; no outcome-dependent tuning.

## Pass 34: safety Camera Viewpoints, batch 2
- Batch: {"n": 128, "original_successes": 38, "edited_successes": 33, "diff_pp": -3.9062, "se_pp": 2.8066, "rescues": 4, "regressions": 9}
- Category cumulative: {"n": 256, "original_successes": 80, "edited_successes": 75, "diff_pp": -1.9531, "se_pp": 1.9531, "rescues": 10, "regressions": 15}
- Candidate unchanged; no outcome-dependent tuning.

## Pass 35: safety Background Textures, batch 1
- Batch: {"n": 128, "original_successes": 80, "edited_successes": 83, "diff_pp": 2.3438, "se_pp": 2.0647, "rescues": 5, "regressions": 2}
- Category cumulative: {"n": 128, "original_successes": 80, "edited_successes": 83, "diff_pp": 2.3438, "se_pp": 2.0647, "rescues": 5, "regressions": 2}
- Candidate unchanged; no outcome-dependent tuning.

## Pass 36: safety Background Textures, batch 2
- Batch: {"n": 128, "original_successes": 79, "edited_successes": 81, "diff_pp": 1.5625, "se_pp": 2.2141, "rescues": 5, "regressions": 3}
- Category cumulative: {"n": 256, "original_successes": 159, "edited_successes": 164, "diff_pp": 1.9531, "se_pp": 1.5109, "rescues": 10, "regressions": 5}
- Candidate unchanged; no outcome-dependent tuning.

## Pass 37: safety Objects Layout, batch 1
- Batch: {"n": 128, "original_successes": 97, "edited_successes": 97, "diff_pp": 0.0, "se_pp": 1.9212, "rescues": 3, "regressions": 3}
- Category cumulative: {"n": 128, "original_successes": 97, "edited_successes": 97, "diff_pp": 0.0, "se_pp": 1.9212, "rescues": 3, "regressions": 3}
- Candidate unchanged; no outcome-dependent tuning.

## Pass 38: safety Objects Layout, batch 2
- Batch: {"n": 128, "original_successes": 96, "edited_successes": 96, "diff_pp": 0.0, "se_pp": 2.4802, "rescues": 5, "regressions": 5}
- Category cumulative: {"n": 256, "original_successes": 193, "edited_successes": 193, "diff_pp": 0.0, "se_pp": 1.5656, "rescues": 8, "regressions": 8}
- Candidate unchanged; no outcome-dependent tuning.

## Safety evaluation complete after pass 38
- All six categories completed: 256 distinct paired variants per category.
- Results: {"Light Conditions": {"n": 256, "original_successes": 237, "edited_successes": 241, "diff_pp": 1.5625, "se_pp": 1.2338, "rescues": 7, "regressions": 3}, "Robot Initial States": {"n": 256, "original_successes": 184, "edited_successes": 176, "diff_pp": -3.125, "se_pp": 1.8253, "rescues": 7, "regressions": 15}, "Sensor Noise": {"n": 256, "original_successes": 162, "edited_successes": 155, "diff_pp": -2.7344, "se_pp": 1.6046, "rescues": 5, "regressions": 12}, "Camera Viewpoints": {"n": 256, "original_successes": 80, "edited_successes": 75, "diff_pp": -1.9531, "se_pp": 1.9531, "rescues": 10, "regressions": 15}, "Background Textures": {"n": 256, "original_successes": 159, "edited_successes": 164, "diff_pp": 1.9531, "se_pp": 1.5109, "rescues": 10, "regressions": 5}, "Objects Layout": {"n": 256, "original_successes": 193, "edited_successes": 193, "diff_pp": 0.0, "se_pp": 1.5656, "rescues": 8, "regressions": 8}}
- Candidate still fails retention: no language gain at seed7; Robot Initial States and Sensor Noise losses exceed 2 pp.
- Final two passes use the frozen 247 language variants at seed19. Combine seeds with uncertainty clustered by variant.

## Pass 39: language replication seed19, batch 1
- {"n": 128, "original_successes": 88, "edited_successes": 90, "diff_pp": 1.5625, "se_pp": 2.2141, "rescues": 5, "regressions": 3}
- Gain is below paired SE. Candidate remains frozen for the last predeclared batch.
