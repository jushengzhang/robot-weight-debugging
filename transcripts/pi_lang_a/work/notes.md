# Language robustness weight-edit study

Constraints: no training, no local model execution; practice variants only.
Forward passes used: 49/50.
Plan: matched-task diagnosis, targeted causal edits with dose response, paired confirmation on disjoint practice variants, non-language safety checks.
Final chosen submission: frozen candidate18, unchanged since pass28. Only 18 instruction-embedding rows replaced with original canonical donor rows. Backup candidate18_weights; recipe candidate18_frozen.json; final_choice.json records evidence.

Pass 1: original, 80 goal language episodes (sample seed 17). 60/80 success. Difficulties 1-3: 34/34; 4:13/15; 5:13/31. Recorded VLM/expert residuals at layers 0,3,6,9,12,15,17 and frames.

Pass 2: original, 82 additional diagnostic episodes. 73/82 success; goal 9/10, spatial22/24, object23/24, long19/24. Diagnostic IDs saved; unused language IDs reserved for confirmation. Hypothesis: missing literal object-name features lead to task confusion, especially in goal suite. Candidate intervention: selected synonym embedding rows, excluding every token occurring in canonical task instructions.

Pass3 attention: language is not absent. L3 heads1/4 attend chiefly BOS; L6 head6 and L9 head4 read newline summary (newline mass roughly0.56-0.62 for plate/stove pair). Synonyms change initial actions despite strong text attention. Causal test A: only space-prefixed basin->bowl and cupboard->cabinet embedding rows, alpha0.5. Both are absent from canonical task-token set; no downstream edits.

Pass4 half-dose probe: basin task action RMS to successful bowl/cabinet reference 0.09044 ->0.03218; all three unexposed inputs bit-identical. Pass5 half-dose rollout, 4 exposed variants x8seeds: original23/32 edited23/32, 2 rescues,2 regressions, diff0.0+/-6.4pp(SE). Basin0/8->2/8; long cupboard/drawer8/8->6/8. Not accepted. Testing full-dose and row ablations.

Pass6 full-dose probe saved. Pass7 full-dose same32 pairs: original19/32 edited23/32 (+12.5pp,SE7.4;5 rescues1 regression), basin0/8->5/8. Null terminations and original long-suite totals vary across runs; do not treat repeat-baseline differences as weight effects. Next: single-row ablations.

Attempted basin-only probe failed with forward-service GPU OOM; did not count. Null rollout records have no instruction/actions and are missing in both arms. Valid basin runs: alpha0:0/5,alpha0.5:2/5,alpha1:5/5. Initial action RMS is not monotone (0.09044,0.03218,0.03768), though both doses improve it.

Second basin-only probe attempt also OOM (not counted). Trying bf16 storage through save_edit to reduce override-transfer memory, with identical intended inference values. Compartment is protected: literal long-suite instructions use it, so no edit to that row.

Pass8 basin-only full substitution: no paired success changes on 8 requested cases (3 missing both arms;4 valid basin episodes all fail). Thus basin row alone is insufficient. Testing cupboard-only next.

Pass9 cupboard-only:1 basin rescue out of4 valid basin cases (3/8 requests missing both arms). Joint row intervention substantially stronger than either alone. Candidate D:7 direct synonym rows at full dose; canonical 74 token IDs protected. Fixed diagnostic panel saved in tune_episodes.json.

Pass10 direct7 candidate: requested126 paired episodes,36 missing both arms; valid summary {'n': 90, 'original': 66, 'edited': 68, 'gain_pp': 2.22, 'se_pp': 1.56, 'rescues': 2, 'regressions': 0}. Two rescues, no regressions. Next candidate adds10 descriptive-cue rows at alpha0.5; ambiguous terms excluded. Exposure-selected diagnostic dose panel saved in pilot_episodes.json.

Pass11 D7 full + descriptive10 half:48 requested pairs,10 missing; valid {'n': 38, 'original': 22, 'edited': 28, 'gain_pp': 15.79, 'se_pp': 5.99, 'rescues': 6, 'regressions': 0}. All6 rescues in goal, no regressions. Saved candidate_half_weights. Testing descriptive full dose on same panel.

Pass12 full17 candidate:48 requested,5 missing; valid {'n': 43, 'original': 25, 'edited': 34, 'gain_pp': 20.93, 'se_pp': 7.1, 'rescues': 10, 'regressions': 1}. Saved candidate_full_weights. Mechanistic control: project each proposed row delta onto the global embedding-mean direction, leaving all orthogonal token information unchanged. No fitting/optimization.

Pass13 common-direction-only control: {'n': 42, 'original': 25, 'edited': 26, 'gain_pp': 2.38, 'se_pp': 2.38, 'rescues': 1, 'regressions': 0} (48 requested). Only1 rescue vs10 for complete substitutions. Now testing orthogonal complement of the common embedding mean direction.

Pass14 orthogonal-complement control: {'n': 48, 'original': 29, 'edited': 38, 'gain_pp': 18.75, 'se_pp': 7.08, 'rescues': 11, 'regressions': 2} (48 requested), with one edited-arm error on an unexposed control. Token-specific directions retain most benefit; common-direction adjustment alone does not. Full17 regression is dairy beverage (milk) incorrectly aliased to cheese. Frozen candidate16 = direct7 + descriptor9 full substitutions, dairy removed. 32768 weight values; all canonical task tokens protected. No VLM blocks, vision, or action weights edited.

Pass15 frozen16 diagnostic: original29/48 edited39/48; +20.83pp(SE5.92),10rescues0regressions. Pass16 matched-input probes saved; analysis in probe16_summary.json. Independent validation manifest generated from unused language IDs before outcomes, up to128/suite; candidate16 remains frozen for these tests.

Pre-validation staging: stageA64/suite (256 total), stageB remaining241 reserved. StageA batches17(goal+long),18(object+spatial). Candidate16 backed up unchanged. Probe16 bowl/plate action RMS to literal0.20231->0.02495; expert L17 RMS6.451->2.216. Cream-cheese description weaker0.27563->0.23912. Unexposed references bit-identical.

Pass17 independent stageA goal64+long64: orig101/128 edited107/128,+4.69pp(SE1.88),6rescues0regressions. Goal47->53/64; long54->54/64. No validation instructions inspected and no candidate changes.

Pass18 stageA object/spatial orig121/128 edit120/128 (-0.78pp SE1.75); discordances audited in validation_safety_summary20.json. Pass19 Camera128 pairs orig84 edit88 (+3.125pp SE1.54), unequal simulator errors. Pass20 InitialStates128 orig94 edit93 (-0.78pp SE0.78), sole regression an edited-arm error. Actual inference-dtype weight audit: exactly16 full embedding-row replacements, zero protected canonical token overlap.

Pass21-24 safety: Light raw112/128 both; Background119->116/128 (3 extra edit-arm errors); Sensor113/128 both; Layout108->109/128. Detailed token/action audit in safety_audit24.json. Keep raw and complete-case metrics separate; infrastructure errors not evidence for/against policy. No safety prompts should read the edited rows.

Pass25 attention maps for two matched paraphrase/literal pairs: summary in attention25_summary.json. Considering only two additional semantically motivated rows (space-prefixed spread->cheese, cup->mug), selected from PRE-validation diagnostics; dairy stays unmodified. New dose panel saved extension_diagnostic_panel.json, candidate16 still active for baseline. StageB remains unused.

Pass26 candidate16 baseline extension-panel80: original49/80 edited55/80,+7.5pp(SE2.96),6rescues0regressions. Now installing candidate18half:16 unchanged full aliases plus spread->cheese and cup->mug at alpha0.5. No training; direct row interpolation only. Aliases exclude every canonical task token.

Pass27 extension18 half-dose on same80: original49 edited61,+15pp(SE4.02),12rescues0regressions; six more successes than base16. Baseline identical across26/27. Now test full-dose18 on identical panel.

Pass28 full18: original48/79 edited64/79,+20.25pp(SE4.55),16rescues0regressions. One successful control(object1079seed89) accidentally omitted. On common79: base16=54,half18=60,full18=64; original48 each. Cheese1379 dose0/8->4/8->8/8. Mug1333 7/8->8/8, ambiguous microwave1466 stays0/8. Frozen18 now; no more lexical selection planned. Fresh confirmation manifest candidate18_confirmation.json contains113+128+113=354 episodes on unused variants, not diagnostic/stageA. StageA256 can be retested with final18 separately. Probe29 request saved. Six-category safety already checked candidate16; final18 needs canonical replay and safety confirmation. No validation instructions inspected for selection.

Pass29 final18 matched-input probe summarized in probe29_summary.json. Confirmation18 now starts on untouched354 variants; no further alias selection planned.

Pass30 FINAL18 fresh confirmation113: original88 edited97,+7.96pp(SE2.86),10rescues1regression; goal29->37/49,long59->60/64. All nine net gains at difficulty5. Weights stay frozen.

Pass31 FINAL18 fresh object/spatial128: original116 edited116,0 discordant outcomes. Fresh confirmation30+31:204->213/241. Next remaining113 entirely unused variants.

Pass32 remaining fresh113: original106 edited106,1rescue1regression. FRESH354 final18 aggregate310->319 (+2.54pp,SE1.01),11rescues2regressions,exact McNemar p0.02246. Independent confirmation criterion met; goal29->37/49. No alias changes. Planned next:33/34 rerun request17/request18 panels with frozen18 (total610 unique non-diagnostic language variants);35-40 repeat six-category safety;41-45 precommitted second-seed replication610 in final18_replication_batches.json; remaining passes for mechanistic controls/final invariance. Candidate18_weights is final intended backup.

Pass33 final18 stageA goal/long:101->108/128,+5.47pp(SE2.3),8rescues1regression. Request audit found a transcription error: long1314seed1003 was replaced by long1315seed1003 (both practice); do not claim exact610 unique coverage until repaired. coverage_repair33.json has missing episode; add it to replication batch5 (98+1 episodes) later. Other30-32 manifests matched. Pass28 missing control already recorded. Next34=request18 panel unchanged; then six safety batches. Frozen18 remains active.

Pass34 final18 object/spatial124->124/128,0discordances. Broad first pass30-34:535->551/610 requested pairs (+2.623pp SE0.762),19rescues3regressions;609 distinct IDs pending repair. D5:63->80/109(+15.596pp). Final18 safety35 Camera101->101/128,36 InitialStates100->100/128;37 Light122->120/128 (-1.563pp SE1.10),zero edited-token exposure. First-input audit of lighting discordances saved light37_input_audit.json. Canonical replay set saved canonical_replay_inputs.json. Full inference-dtype audit:exactly18 changed rows, no protected-token intersection.

Pass38 Background: {"n": 128, "original": 122, "edited": 121, "gain_pp": -0.781, "se_pp": 0.781, "rescues": 0, "regressions": 1, "unpaired": 0, "termination": {"original": {"success": 122, "max_steps": 6}, "edited": {"success": 121, "max_steps": 7}}}
Pass39 Sensor Noise: {"n": 128, "original": 109, "edited": 108, "gain_pp": -0.781, "se_pp": 0.781, "rescues": 0, "regressions": 1, "unpaired": 0, "termination": {"original": {"success": 109, "max_steps": 19}, "edited": {"success": 108, "max_steps": 20}}}
Pass40 Objects Layout102->102/128, no discordances. Frozen18 unchanged. Requests41-45 saved exactly from precommitted replication manifest; 45 additionally contains the separately labelled pass33 coverage repair. All final18 six-category safety summaries in final18_safety40_summary.json.

Pass41 frozen18 second-seed replication: {"n": 128, "original": 108, "edited": 112, "gain_pp": 3.125, "se_pp": 1.901, "rescues": 5, "regressions": 1, "unpaired": 0}. Manifest exact; no weight changes.

Pass42 frozen18 second-seed replication: {"n": 128, "original": 111, "edited": 120, "gain_pp": 7.031, "se_pp": 2.269, "rescues": 9, "regressions": 0, "unpaired": 0}. Manifest exact; no weight changes.

Pass43 frozen18 second-seed replication: {"n": 128, "original": 112, "edited": 112, "gain_pp": 0.0, "se_pp": 1.569, "rescues": 2, "regressions": 2, "unpaired": 0}. Manifest exact; no weight changes.

Pass44 frozen18 second-seed replication: {"n": 128, "original": 113, "edited": 116, "gain_pp": 2.344, "se_pp": 1.342, "rescues": 3, "regressions": 0, "unpaired": 0}. Manifest exact; no weight changes.

Pass45 completed second-seed replication and repaired pass33 coverage. Manifest-aligned first seed: {"n": 610, "original": 535, "edited": 551, "gain_pp": 2.623, "se_pp": 0.762, "rescues": 19, "regressions": 3, "unpaired": 0}. Second seed: {"n": 610, "original": 533, "edited": 553, "gain_pp": 3.279, "se_pp": 0.826, "rescues": 23, "regressions": 3, "unpaired": 0}. Combined variant-cluster SE(pp): 0.7146. Full audit in final18_evidence45.json. Candidate18 stays unchanged. Canonical fixed-input request46 prepared for all40 tasks.

Final tests precommitted before outcomes:46 canonical40 fixed-input replay; 47/48 all204 non-diagnostic D4/D5 variants at third seeds7000+ (secondary, not pooled with primary broad result);49 fresh128Light;50 fresh64Background+64Sensor. No unused practice language variants remain. Full weight audit final_weight_audit45.json confirms exactly18 donor-row substitutions and zero canonical-token overlap.

Pass46 canonical fixed-input replay: {"inputs": 40, "actions_exact": 40, "all_recorded_activations_exact": 40, "max_action_abs_difference": 0.0, "max_activation_abs_difference": 0.0, "activation_arrays_compared": 1920, "nonidentical_inputs": []}. Frozen18 unchanged.

Pass47 Third-seed D4/D5 check, first half: {"n": 102, "original": 74, "edited": 82, "gain_pp": 7.843, "se_pp": 2.675, "rescues": 8, "regressions": 0, "unpaired": 0}. Manifest exact; frozen18 unchanged.

Pass48 completes all204 hard-language third-seed pairs: {"n": 204, "original": 144, "edited": 158, "gain_pp": 6.863, "se_pp": 2.028, "rescues": 16, "regressions": 2, "unpaired": 0}. By difficulty: {"4": {"n": 95, "original": 87, "edited": 86, "gain_pp": -1.053, "se_pp": 1.83, "rescues": 1, "regressions": 2, "unpaired": 0}, "5": {"n": 109, "original": 57, "edited": 72, "gain_pp": 13.761, "se_pp": 3.315, "rescues": 15, "regressions": 0, "unpaired": 0}}. This is a secondary difficulty-enriched check, not pooled into the primary1220-pair estimate. Frozen18 unchanged. Final two passes are safety only.

Pass49 fresh Light120->122/128 (+1.563pp;3rescues1regression), zero source-token exposure. Combined final18 Light242->242/256. Manifest exact. Submission checksum reverified before last pass. Primary language:1068->1104/1220 (+2.951pp, variant-cluster SE0.7146), 42rescues6regressions,610 non-diagnostic variants x2seeds. Primary D5:127->159/218 (+14.679pp). Independent fresh354 initial confirmation:+2.54pp(SE1.01). Secondary all204 D4/D5 at third seeds:144->158 (+6.863pp,SE2.028); D5:57->72/109. Canonical replay40/40 action chunks and1920/1920 activation arrays exact. Final pass50 precommitted64freshBackground+64freshSensor; retain all outcomes. Active weights are the chosen final18; no subsequent edits planned. See roll_50.jsonl for the final safety result when available.
