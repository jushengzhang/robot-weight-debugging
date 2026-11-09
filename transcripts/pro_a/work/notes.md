# Robustness weight-edit study
Constraints: direct, mechanistically motivated weight edits only; no training, no local model execution.
Forward budget: 40. Starting with matched diagnosis and a predefined practice validation partition.
All comparisons will use fixed episode seeds; final checks include original/object/semantic.

## Pass 1: baseline (128 episodes, sampler seed 1701)
Environment 17/25, object 23/26, position 3/26, semantic 21/24, task 5/27; 13 task failures reach original goal; 28/59 failures have regrasp loops. Sampler ignored 'original' (not in provided practice file). Original-task testing currently unavailable.
Partition by SHA256(suite/task/init_idx) mod 5: 0-2 development; 3 validation; 4 final, excluding baseline episodes from validation/final.
Rounded aggregate resid norm display was similar across layers, but exact equality checks were FALSE. Do not infer localization from the aggregate display; use per-input probes.
Concrete old-goal examples: task1946 BBQ request moves tomato sauce 0.359m, BBQ 0; task1562 tomato request moves salad dressing 0.352m, tomato 0.

## Passes 2-3: diagnostic probes
Expert L9H4 attends final newline 47-63% across four first observations; L9H2 attends wrist patch (13,3) 75-81%, total wrist 98.6-99.5%. Other language-heavy heads often attend BOS, so total language mass is not content attention. Position1896/1898 move milk 0.353/0.376m, tomato unchanged; trajectories mimic familiar tomato reach. Probe residuals are modality means, not token-resolved. Final VLM layer output is not itself cached, so large L17 residual norms do not identify a causal fault.
Candidate A: only expert L9H4 o_proj columns 1024:1280 multiplied by 1.5. Hypothesis: strengthen instruction summary versus scene habitual action; no global scaling.

## Pass 4: candidate A, L9H4 output x1.5, dev128
Original 69/128, edit69/128; paired 0.0 +/-2.5pp SE, 5 wins/5 losses. Task5->6, position3->4, environment17->13, object23->23, semantic21->23. Original-goal task failures13->12. REJECTED (no gain, environment damage).
Candidate B: reset; L9H4 q_proj rows1024:1280 x1.5 only. Hypothesis: sharpen instruction-vs-visual selection, not output magnitude.

## Passes 5-6: candidate B, L9H4 query x1.5
Probe verifies newline attention1946 47.2->56.1%, but rollout dev128 69->65 successes; paired -3.1 +/-2.2pp; task5->4, position3->3, environment17->13, object23->22, semantic21->23. REJECTED. More final-token attention alone is not a repair.
Candidate C: reset; expert L9H2 o_proj columns512:768 x0.5, a causal attenuation of fixed wrist hotspot pathway.

## Pass 7: candidate C, L9H2 output x0.5
dev128 69->69; paired0.0 +/-2.2pp SE, 4 wins/4 losses; task5->5, position3->3, environment17->15, object23->23, semantic21->23. Old-goal failures13->11. REJECTED.
Next: controlled prompt-only replay contrasts. Copies of saved practice observations/noise, replace prompt with that record's instruction_orig. No new simulation states and no training. Use differences to identify instruction-sensitive output subspaces rather than whole-head gains.
Counterfactual probe REFUSED (did not count): tool permits only exact rollout inputs. No workaround attempted. Switching to listed practice episodes matched on scene/init_idx across all five ptypes. Will check actual image equality before interpreting instruction contrasts.

## Pass 8: matched baseline80 (16 groups, all five ptypes)
Task5/16 (10/11 failures old goal), position6/16, environment9/16, object13/16, semantic16/16. Exact initial image-matched task/semantic groups: [].

## Pass 9: early attention maps
Layers0-2 dominated by BOS; first appreciable content attention at L4: H3 words25-32%, H7 words26-35%, versus ~0 at L0. Initial task/semantic image differences are nonzero for all matched groups, so no pure prompt contrast is available.
Candidate D: only L4H3/H7 output columns x1.5, testing early content reading vs failed late newline amplification.

## Pass 10: candidate D, L4H3/H7 output x1.5
dev128 69->68; paired -0.8 +/-1.8pp SE; task/position unchanged; environment17->15, semantic21->22, object unchanged. REJECTED.
Four head-gain variants gave no meaningful target gains. Pivot to upstream visual scene/location encoding.

## Weight-localized upstream candidates
Projector top singular value15.084 vs second3.689; top mode12.88% of matrix energy, left vector dominated by dims674(0.733),604(0.279),738(0.172). Pooled visual input score mean-67.10, SD2.11 across128 scenes, largely common-mode. Candidate E: projector W <- W -0.2*s0*u0*v0^T (only top singular mode attenuated).
Position embedding outlier rows193(norm313.8),47(231.1),48(198.1),243(97.3),13(93.9), median18.24. Patch193 is a recurring base-camera attention hotspot; wrist211 is NOT a positional outlier. Prepare independent outlier-residual attenuation preserving local positional neighborhood; not installed yet.

## Pass 11: candidate E, projector top mode retain80%
dev128 69->72; +2.3 +/-2.3pp SE (6 wins/3 losses). Task5,position3,environment17,object23 unchanged; semantic21->24. Not a robust-target gain; not retained as best.
Candidate F: reset; only five SigLIP position outlier rows, new_row=neighbor_mean+0.5*(original_row-neighbor_mean). All other rows unchanged; preserves local reference while halving non-geometric exceptional residuals.

## Pass 12: candidate F, positional outlier residual retain50%
dev128 69->74, +3.9 +/-2.0pp SE, 6 wins/1 loss. Task5->7, environment17->18, position3 unchanged, object23 unchanged, semantic21->23; loops29->24, old-goal failures13->12. Provisional only, not independent validation. Saved provisional_register050. Next dose retain20%, same five rows, no other edits.

## Passes 13-14: positional residual retain20%, five outliers
Probe: L9H4 base193 attention task1946 13.23->0.11%, position1896 28.90->0.40%; wrist211 largely preserved (77.57->72.50% task;77.36->75.17% position). Dev12869->71, +1.6 +/-2.7pp SE, 7 wins/5 losses. Task5->8, position3->3, environment17->16, object23->23, semantic21->21. Task dose-response 5(original),7(retain50%),8(retain20%), but stronger dose harms other types. Next isolate row193 retain20%, all other rows restored.

## Pass 15: row193 only, positional residual retain20%
Dev12869->73, +3.1 +/-2.5pp SE, 7 wins/3 losses. Task5->8, position3 unchanged, environment17 unchanged, object23->22, semantic21->23; old-goal task failures13->9. Still provisional; no independent validation yet.

Validation128_a fixed with RNG161601, balanced as development; unused IDs from the predeclared validation partition, seed7. Testing pass15's exact row193-only edit, before any combinations. Outlier residual cosine: row193 vs other four 0.26-0.34; other four mutually 0.73-0.90, supporting separate pathways.

## Pass16: independent validation of row193 keep20%
82->79/128, -2.3 +/-2.6pp, 4 wins/7 losses. Environment13->15, object24->21, position11->9, semantic22->21, task12->13. Old-goal failures6 unchanged. REJECT row193-only candidate: development gain did not generalize.

Candidate G: reset position edits; projector top singular mode retain50% (earlier retain80% gave only semantic gain). All other singular modes unchanged. Test dose-response of common visual carrier, not overall visual scaling.

## Pass17: projector top mode retain50%
Dev69->64/128, -3.9 +/-2.6pp, 3 wins/8 losses. Environment17->11, object23->21, position3 unchanged, semantic21->24, task5 unchanged. REJECT: common projector carrier is important; removing more does not repair target failures.

## Pass18: late attention diagnosis
Initial expert L14-16 has strong wrist access, L16H3 switches from wrist253 to early prompt token at grasp stage; prefix mass falls for many L17 heads (action-token attention omitted from maps). Not simple language blindness. VLM residuals dominated by channels674/50 early, late1577, but no evidence yet that these essential carrier channels should be erased.
Restoring exact five-row keep50% candidate for validation128_a.

## Pass19: five-row keep50% on validation128_a
82->82/128, 0.0 +/-2.5pp; 5 wins/5 losses. Env13->16, object24->23, position11->8, semantic22->23, task12 unchanged. Loops27->20. Not a validated overall improvement; position damage. Set aside.

Refined noun-content attention (excluding BOS/newline and generic syntax): L12H2 strongest mean30.2%, but task1946 only7.1%; L12H6 explicitly reads BBQ10.1% while BOS54.5%. Earlier L9H4 and L4 head boosts were not referent-specific. Plan an analytic contrast-subspace edit, using matched task/semantic initial observations (not identical images; visual contrasts will be measured and projected away). No success labels, regression, gradients, or loss fitting.

## Pass20: matched initial-state contrast probe
32 exact replay inputs, 16 task/semantic groups, VLM residual means before cache layers4/6/9/12/15. Candidate H: at L12, map normalized saved residual directions through existing RMSNorm/value weight geometry; remove top4 visual summary directions from paired language contrasts, take top4 contrast directions. Only expert L12H2/H6 output matrices changed: O_h <- O_h +0.5*(O_h Q)Q^T. No fitted targets or losses; this is a rank4 semantic-direction gain, not whole-head scaling. Mean summaries and nonidentical images limit the causal interpretation until rollout testing.

## Pass21: L12 semantic-subspace gain1.5
Dev69->68/128, -0.8 +/-2.1pp, 3 wins/4 losses. Env17->15, object23->22, position3 and task5 unchanged, semantic21->23. Old-goal failures13->12, loops29->35. REJECT current semantic-value gain.

Candidate I: expert L14H0/H1 o_proj columns0:512 x1.5; all other weights original. Probe18: wrist attention initial tasks64-81%, grasp-stage position/object81-99%; top locations differ (w193/w114 initial vs w254/w242 grasp). Hypothesis: increase current visual feedback for grasp correction; not gripper/action scaling. Stage comparisons are not pure intervention evidence; test rollouts and matched-stage maps next.

## Pass22: L14H0/H1 output gain1.5
Dev69->72/128, +2.3 +/-2.1pp, 5 wins/2 losses. Env17->16, object23 unchanged, position3->4, semantic21->24, task5 unchanged. Loops29->28, old-goal13 unchanged. Weak evidence for the proposed feedback mechanism. Confirmed sample n128 seed1701 reproduces exact development IDs/seeds.

Control requested with no edits was refused without consuming a pass. Saved bitwise-original position_embedding.weight as a no-op control; load vs load_original equality verified (0 changed elements).

## Pass23: no-op control
Saved position tensor exactly equals original. Dev69->69/128, 0 discordant outcomes. Secondary metrics not bitwise stable: mean success steps154.6 vs154.7 and loops30 vs28. Interpret success-pair comparisons primarily; repeated semantic rescues are not unique mechanistic evidence for individual edits.

Candidate J: same matched contrast geometry, now key space at VLM L12. Remove top4 pooled visual key directions from language contrasts; take top4 residual directions Qk. K <- K + Qk Qk^T K (2x in rank4 subspace only). Tests semantic selection rather than head/value gain; all other weights original.

## Pass24: L12 semantic-key gain2
Dev69->69/128, 0.0 +/-1.9pp, 3 wins/3 losses. Task5/position3 unchanged, old-goal failures13 unchanged; env17->16, object23->22, semantic21->23. REJECT. Mean-level semantic key/value subspaces did not repair binding.

Candidate K: early lexical-contrast edit. Fixed physical referent/spatial/goal-predicate vocabulary (no outcome labels). For its 103 embedding rows, E_i <- E_i +0.5*(E_i - mean_function_word_embedding). All other rows original. This changes content direction as well as residual magnitude, so it is not canceled by the first RMSNorm and is not whole-tensor scaling. Hypothesis: preserve requested referent signal before scene mixing. Dose1.5; test task following and already-solved categories.

## Pass25: lexical contrast gain1.5
Dev69->70/128, +0.8 +/-2.1pp, 4 wins/3 losses. Env17->16, object23 unchanged, position3->2, semantic21->22, task5->7. Old-goal failures13 unchanged. Testing dose2.0 on the same 103 rows before judging the mechanism.

## Pass26: lexical contrast gain2
Dev69->71/128, +1.6 +/-2.7pp, 7 wins/5 losses. Env17->15, object23 unchanged, position3->5, semantic21->23, task5 unchanged. Old-goal failures13->14. No monotonic task dose-response; reject lexical candidates as established instruction-binding repairs.

Candidate L: retain50% exceptional residual only in position rows13,47,48,243; restore row193 exactly. These four residuals have mutual cosine0.73-0.90, versus0.26-0.34 with row193. Prior five-row keep50% env gains dev+1/25,val+3/25 but valposition-3/26; isolate coherent four-row component to test that tradeoff. All other weights original.

## Pass27: four aligned outliers keep50%, row193 original
Dev69->69/128, 0.0 +/-2.7pp, 6 wins/6 losses. Env17->14, object23 unchanged, position3->4, semantic21->24, task5->4. REJECT. This ablation indicates the earlier five-row benefits require row193; the other four are not independently beneficial for environment robustness.

Candidate M: row193 only at exact conservative keep50% residual; other255 rows original. Distinct from pass15/16 isolated keep20% dose (which failed validation). Provides localization and dose comparison to full five-row and four-row ablations.

## Pass28: row193 only keep50%
Dev69->70/128, +0.8 +/-2.4pp, 5 wins/4 losses. Env17->14, object23/position3 unchanged, semantic21->23, task5->7; old-goal13->11. Not a reliable overall repair. Five-row benefits are non-additive; isolated components both damage environment.

Candidate N: local spatial residual smoothing on the251 non-register rows only. P_i <- 0.5*P_i+0.5*mean(neighbor rows excluding registers). All5 register rows remain exactly original. One-hop averaging approximately preserves smooth coordinate trends while reducing patch-specific irregularities; tests fine absolute-location shortcuts separately from global registers. No image changes or inference changes.

## Pass29: ordinary-row spatial smoothing50%
Dev69->71/128, +1.6 +/-1.9pp, 4 wins/2 losses. Env17/object23/task5 unchanged, position3->4, semantic21->22. Old-goal failures13->15; no established instruction benefit.

Candidate O (last new mechanism combination): ordinary local residual smoothing50% plus exact five-register residual retain50% from pass12. One position table only. Test complementarity of spatial regularization and global register correction, then stop exploratory edits and reserve remaining passes for paired validation/safety.

## Pass30: combined positional repair
Dev69->73/128, +3.1 +/-2.7pp, 8 wins/4 losses. Env17->16, object23/position3 unchanged, semantic21->24, task5->7, old-goal13->11. Not clearly better than sparse five-row edit.

## Predeclared closing plan (before final results)
Pass31: combined edit on existing validation128_a. Choose combined only if validation successes>=85 (sparse five-row=82), object>=23 and semantic>=22; otherwise choose sparse five-row keep50% for its smaller footprint. No more parameter tuning after this comparison. Reserve six balanced, unused final-partition batches (768 total) for paired confirmation, plus two unused validation-partition safety batches of64 object+64 semantic (256 total). Keep a nontrivial edit only if independent final overall gain exceeds1.96 paired SE and safety object/semantic combined drop is no worse than2pp. If not, revert to original rather than submit an unsupported repair. Original LIBERO episodes are absent from the allowed practice list; no claim of measured original-task non-inferiority can be made.

## Pass31 and freeze
Combined edit validation82->83/128, +0.8 +/-2.6pp, 6 wins/5 losses. Env13->14, object24->23, position11->10, semantic22->24, task12 unchanged. Does not meet predeclared selection threshold. Freeze sparse five-row keep50% exactly as pass12/19. Frozen hash 44d71c5e3051dd09cd7473497428e25aa5194cef76caf0a0cedf49d3910f0916. No further tuning; six untouched final batches plus two safety batches next.

## Pass32: frozen edit, final batch1
77->73/128; -3.1 +/-2.5pp paired SE; 3 wins/7 losses. Env11->10, object22->23, position8->7, semantic24->21, task12 unchanged. Old-goal failures9->10. Frozen edit unchanged; all six batches will be pooled before the predeclared keep/revert decision. Planned 1024 IDs are distinct.

## Pass33: frozen edit, final batch2
66->69/128; +2.3 +/-2.1pp paired SE; 5 wins/2 losses. Env12->11, object22 unchanged, position4->7, semantic23 unchanged, task5->6. Pooled batches1-2: {"n": 256, "original": 143, "edited": 142, "diff_pp": -0.391, "se_pp": 1.614, "wins": 8, "losses": 9}. Audit confirms no final/safety ID appeared in passes1-31; no edits changed.

## Pass34: frozen edit, final batch3
81->84/128; +2.3 +/-2.6pp paired SE; 7 wins/4 losses. Env17->20, object20->21, position9->6, semantic24->25, task11->12. Pooled batches1-3: {"n": 384, "original": 224, "edited": 226, "diff_pp": 0.521, "se_pp": 1.38, "wins": 15, "losses": 13}.

## Remaining-pass allocation
Passes35-37: final batches4-6; passes38-39: preselected safety batches1-2. Pass40: exact saved-input attention probe of the same frozen five-row keep50% edit (no tuning; diagnostic only, excluded from confirmation statistics), then apply the predeclared keep/revert rule. No new candidate will be introduced.

## Pass35: frozen edit, final batch4
69->66/128; -2.3 +/-2.8pp paired SE; 5 wins/8 losses. Env14->12, object19->21, position7->6, semantic23 unchanged, task6->4. Pooled batches1-4: {"n": 512, "original": 293, "edited": 292, "diff_pp": -0.195, "se_pp": 1.252, "wins": 20, "losses": 21}. Frozen hash reverified; no changes.

## Passes36-37: frozen edit, final batches5-6
Batch5:68->70/128, +1.6 +/-2.7pp; 7 wins/5 losses. Batch6:73->71/128, -1.6 +/-2.5pp; 4 wins/6 losses.

## Completed independent confirmation (768 episodes)
{"overall": {"n": 768, "original": 434, "edited": 433, "diff_pp": -0.13, "se_pp": 1.034, "wins": 31, "losses": 32}, "approx_95pct_ci_pp": [-2.157, 1.897], "confirmation_passed": false, "by_ptype": {"environment": {"n": 156, "original": 85, "edited": 83, "diff_pp": -1.282, "se_pp": 2.57, "wins": 7, "losses": 9}, "object": {"n": 150, "original": 123, "edited": 130, "diff_pp": 4.667, "se_pp": 2.381, "wins": 10, "losses": 3}, "position": {"n": 156, "original": 37, "edited": 35, "diff_pp": -1.282, "se_pp": 2.031, "wins": 4, "losses": 6}, "semantic": {"n": 150, "original": 142, "edited": 139, "diff_pp": -2.0, "se_pp": 2.406, "wins": 5, "losses": 8}, "task": {"n": 156, "original": 47, "edited": 46, "diff_pp": -0.641, "se_pp": 2.132, "wins": 5, "losses": 6}}, "by_suite": {"libero_10": {"n": 190, "original": 91, "edited": 89, "diff_pp": -1.053, "se_pp": 2.109, "wins": 7, "losses": 9}, "libero_goal": {"n": 195, "original": 121, "edited": 120, "diff_pp": -0.513, "se_pp": 2.465, "wins": 11, "losses": 12}, "libero_object": {"n": 196, "original": 111, "edited": 116, "diff_pp": 2.551, "se_pp": 1.687, "wins": 8, "losses": 3}, "libero_spatial": {"n": 187, "original": 111, "edited": 108, "diff_pp": -1.604, "se_pp": 1.93, "wins": 5, "losses": 8}}}
The confirmation requirement FAILED. Under the frozen decision rule, restore original weights after finishing the already planned safety batches and diagnostic probe; safety results cannot rescue a failure of the overall-gain requirement. Candidate remains temporarily installed only for those remaining checks. Allowed practice list contains no original/unperturbed ptype.

## Passes38-39: independent safety checks (256 episodes)
Batch1:118->117/128, 3 wins/4 losses. Batch2:119->117/128, 3 wins/5 losses.
{"overall": {"n": 256, "original": 237, "edited": 234, "diff_pp": -1.172, "se_pp": 1.514, "wins": 6, "losses": 9}, "approx_95pct_ci_pp": [-4.139, 1.796], "observed_drop_within_2pp": true, "by_ptype": {"object": {"n": 128, "original": 114, "edited": 111, "diff_pp": -2.344, "se_pp": 2.344, "wins": 3, "losses": 6}, "semantic": {"n": 128, "original": 123, "edited": 123, "diff_pp": 0.0, "se_pp": 1.921, "wins": 3, "losses": 3}}}
Observed combined safety drop is within2pp, but uncertainty does not establish 2pp non-inferiority. Confirmation-gain criterion failed; final decision is REVERT. Only the already planned diagnostic probe remains before restoration.
