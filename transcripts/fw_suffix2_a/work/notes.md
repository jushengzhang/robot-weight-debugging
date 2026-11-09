# New session: FastWAM suffix invariance
## Constraints and prior findings
50 forward passes; no training/local network execution. Only practice metadata read.
Earlier mean-contrast gates and projections failed to restore clean actions. Prior candidate is NOT applied.
Prior notes report +~1.8pp tagged with weak significance and >2pp paraphrase loss; not a baseline to retain automatically.
## Splits
splits.json: fixed balanced diagnostic48 tagged+16 language; development96 tagged+48 language;
six disjoint tagged validation sets of128, four language validation sets of128. Selection independent of outcomes.
## Plan
Matched raw/strip diagnostics, improve token-level attribution rather than relying only on context means;
causal expert/pathway and dose comparisons; freeze and validate paired success and safety.
## Passes
None yet.
## Current edit
Original (no edits).

### Passes1-2: matched baseline
48 tagged diagnostic: raw26/48=54.17%, strip36/48=75%; +20.83pp. Language16:10/16 both. Camera1->4/8; goal suite7->12/16 including language. Inputs and episode attn summaries saved.

### Pass3 reference and temporary instrumentation
56 inputs:24 raw/clean pairs+8 paraphrases; noise_seed0. Probe exposes action residual means, not token embeddings/cross-text maps. Temporary block0 uniform readout measures old detector score hinge moments at1023 thresholds [-4,12]. MUST REMOVE instrumentation before repair evaluation.

### Pass4 detector histogram
At u>.5 raw tail mass~.156 vs clean/paraphrase~0. Camera/initial/noise~.281; background/light~.042; layout~.011. At u>1 raw~.089 and almost no background/light/layout. Old detector incompletely covers tags. Histogram removed.
Temporary 30-band context readout now installed (action only): near-linear896 coordinate slices, exact u and GELU hinges; constant anchored q/k select bands at30 centres. Five slices planned. MUST CLEAR before candidate.

### Passes5-6 band context acquisition
Slices0:896 and896:1792 saved; readout time variation exactly0; nontrivial band-specific contexts. Temporary instrumentation advances to1792:2688.

### Passes7-9 complete context-band acquisition
context_bands_56x30x4096.npy saved (24raw+24clean+8language). Readout time variation0 for all slices. ALL426 instrumentation edits removed. Subsequent analysis uses observed vectors only.

### Candidate prefix-band rank8
74 matched raw/clean band pairs with cosine>.65,u<.45,norm>1. Clean span protects256 leading clean/paraphrase band PCs plus exact9D proprio span. SVD of clean-orthogonal contrasts defines P=I-BU^T, U^TB=I. Both text_embedding.0.weight edited; no optimization or network execution in kernel. Awaiting probe10.

Probe metrics: {"pass": 10, "baseline_gap": 0.06999068294422646, "gap": 0.07682491736866233, "ratio": 1.097644916965332, "clean_drift": 0.0026842525812204405, "language_drift": 0.002243982983163816, "raw_change": 0.028053576132496826, "correction_fraction": -0.02208435721695423, "category_ratio": {"Background Textures": 0.9727, "Camera Viewpoints": 1.0317, "Light Conditions": 1.4042, "Objects Layout": 0.9666, "Robot Initial States": 1.0442, "Sensor Noise": 1.0159}}

### New candidate: five-prototype context gate
Four category-balanced clean-orthogonal tag prototypes + numeric-low prototype, derived from band vectors (not context means). Threshold.6,gain64,strength12,key64,both experts. Five low-output-energy hidden neurons and one text output coordinate/expert carry detector sum; one q/k coordinate/head gates it. Self-attention/MLP/action head unchanged. Probe11 pending. Prefix rank8 rejected (gap ratio1.0976).

Probe metrics: {"pass": 11, "baseline_gap": 0.06999068294422646, "gap": 0.10863939890584676, "ratio": 1.5521980117327665, "clean_drift": 0.0022938696400617888, "language_drift": 0.0013765289590682503, "raw_change": 0.0652260734962189, "correction_fraction": -0.27041712403297424, "category_ratio": {"Background Textures": 0.8796, "Camera Viewpoints": 2.5509, "Light Conditions": 0.9985, "Objects Layout": 0.9621, "Robot Initial States": 1.6044, "Sensor Noise": 1.3622}}

### Pass11 five-prototype gate rejected
Gap ratio1.5522; clean drift.00229, language.00138. Numeric categories worsened (Camera2.55x); background improved. Indicates low-score novelty is not safe to mask wholesale.
Next: both-expert clean-language manifold projection, rank256 clean PCs+exact proprio span (2 text entry tensors). Removes input innovations while retaining their task-like components, unlike token gating. Probe12 pending.

Probe metrics: {"pass": 12, "baseline_gap": 0.06999068294422646, "gap": 0.09432202098147915, "ratio": 1.3476368141262691, "clean_drift": 0.06527493768555401, "language_drift": 0.1017880927618637, "raw_change": 0.04335965480658194, "correction_fraction": -0.2161685824394226, "category_ratio": {"Background Textures": 1.4249, "Camera Viewpoints": 1.1434, "Light Conditions": 1.4402, "Objects Layout": 2.4154, "Robot Initial States": 1.4423, "Sensor Noise": 1.0504}}

### Pass12 clean manifold rejected
Gap ratio1.3476, clean drift.0653, paraphrase.1018. The30 prior-detector bands miss useful individual-token directions even though they preserve band means. All edits removed.
Plan refined acquisition:300 selectors via30 layers x10 diffusion-step-dependent fixed selectors, using a clean-PC direction to separate ordinary words. Measurement only; final candidate must be independently validated.

### Restart after pass12
Failed dense readout had left10 partial instrumentation edits; all cleared. Reconstructed scanner with targeted timestep rows only, 896-coordinate readout, 30x10 selectors along clean PC0. Temporary instrumentation now installed for slice0:896; MUST CLEAR before candidate evaluation.

### Pass13 dense context scan
First896 coordinates acquired on56 registered inputs. Dense readout uses scaled cleanPC1 and targeted timestep signal. Slice896:1792 now installed; instrumentation only.

Passes14-15: dense slices896:1792 and1792:2688 saved; finite readouts. Next2688:3584. All522 current edits are instrumentation, not a repair.

### Passes16-17 dense acquisition
Fourth/fifth slices saved. Pass17 request accidentally used clean paths at tagged rows16:24; must replace these eight final512-coordinate readouts with pass18 before assembly. Instrumentation still active.

### Pass18 correction and acquisition complete
Eight last-slice rows corrected. context_dense_56x300x4096.npy assembled. ALL522 instrumentation edits removed. Dense scanner observations available for static geometry analysis; no repair currently applied.

### Candidate dense prefix-mean rank16
182 mutual raw/clean context matches across24 instructions; per-instruction contrast means define16D oblique projector. Protect512 dense clean/paraphrase PCs plus exact9D proprio span. Only two text_embedding.0.weight tensors changed. Probe19 pending.

Probe metrics: {"passno": 19, "baseline_gap": 0.06999068294422646, "gap": 0.07035426372918603, "ratio": 1.0051947026327674, "clean_drift": 0.0010507095571341905, "language_drift": 0.0013568638310521275, "raw_change": 0.0034461505285277405, "correction_fraction": -0.00399604020640254, "edited_gap": 0.07033336637344725, "category_ratio": {"Background Textures": 1.0214, "Camera Viewpoints": 0.9763, "Light Conditions": 0.9977, "Objects Layout": 0.9663, "Robot Initial States": 1.021, "Sensor Noise": 1.0101}}

Pass19 dense prefix rank16 rejected: gap ratio1.0052, clean drift.00105, paraphrase.00136. Matched-prefix contrast removal has little causal effect; all edits cleared. Next selective unmatched-token gate protects raw prefix as well as clean text.

### Candidate dense unmatched-context gate
Six category-mean novel-feature directions; protect512 clean PCs+256 raw-prefix PCs+proprio. Threshold.20,gain128,prior probe11 q/k gate wiring strength12,key64,both experts. Measured max clean detector.017, max matched-prefix.086, unmatched median max.349. Probe20 pending.

Probe metrics: {"passno": 20, "baseline_gap": 0.06999068294422646, "gap": 0.08056094334686549, "ratio": 1.151023821428663, "clean_drift": 0.00697770662584063, "language_drift": 0.0014439027691692807, "raw_change": 0.0244613895530597, "correction_fraction": -0.10135454684495926, "edited_gap": 0.07716231058509426, "category_ratio": {"Background Textures": 0.9094, "Camera Viewpoints": 0.816, "Light Conditions": 0.9939, "Objects Layout": 1.0064, "Robot Initial States": 1.5517, "Sensor Noise": 0.9798}}

Pass20 unmatched gate: gap ratio1.151 overall, Camera.816, Initial1.552, clean drift.00698, language.00144. Probe is mixed, not a retention basis. Paired independent development96 rollout planned before further gate tuning.

### Pass21 development rollout, dense six-detector both-expert gate
96 paired:47->52 successes, +5.21pp, SE3.4pp,8 gains/3 losses. Background+2,Camera0,Light-2,Layout+1,Initial+3,Noise+1 episodes. Not conclusive. Immutable copy candidate_dense_gate_t020_both. Video-only causal ablation now active for pass22.

### Pass22 video-only causal test
Same96 development episodes:47->52 successes,+5.21pp,SE3.1pp,7 gains/2 losses. Indirect video route alone reproduces total gain with fewer regressions. Action-only now installed for pass23. candidate_helpers.py restores frozen gate, changes only expert selection/dose/threshold, and computes paired statistics.

### Pass23 action-only causal test
Same96 development:47->48,+1.04pp,SE2.3pp,3 gains/2 losses. Video-only is more promising than direct action-only. Video gate penalty half-dose now installed for pass24; detector unchanged.

### Pass24 video gate half-dose
96 paired:47->50,+3.125pp,SE2.8pp,5 gains/2 losses. Original dose pass22 was+5.21pp. Double-dose video gate now active for pass25; detector threshold.20 unchanged.

### Pass25 video gate double-dose
{"passes": [25], "n": 96, "original_success": 47, "edited_success": 47, "wrong_to_right": 5, "right_to_wrong": 5, "diff_pp": 0.0, "se_pp": 3.3113308926626095, "ci95_pp": [-6.490208549618714, 6.490208549618714]}
Double-dose rejected; standard dose remains development leader, not validated.

### Candidate video-only threshold .10, standard dose
Lower threshold covers79.2% vs65.7% of diagnostic novel contexts; max observed clean score.01784, matched-prefix.08563. No new directions or other weight changes. Development rollout pending.

### Pass26 lower-threshold video gate rejected
{"passes": [26], "n": 96, "original_success": 47, "edited_success": 45, "wrong_to_right": 4, "right_to_wrong": 6, "diff_pp": -2.083333333333333, "se_pp": 3.3044250855782513, "ci95_pp": [-8.560006501066706, 4.39333983440004]}
Restored video-only threshold.20 standard dose. Next: paired strip safety on80 tagged development+48 language development variants.

### Pass27 standard video-gate development safety FAILED
{"passes": [27], "n": 128, "original_success": 90, "edited_success": 84, "wrong_to_right": 2, "right_to_wrong": 8, "diff_pp": -4.6875, "se_pp": 2.445109764777652, "ci95_pp": [-9.479915138964198, 0.10491513896419757]}
Clean60->57/80 (-3.75pp); paraphrase30->27/48(-6.25pp). Do not retain this gate without correcting collateral damage and independent validation.

### Null-carrier video gate candidate
Static QR gives a carrier orthogonal to all retained text-MLP output columns and its bias after repurposing6 hidden neurons. No ordinary output channel discarded. Carrier projected out of ordinary keys/values; only dedicated gate key rows read it. Same six detectors,threshold.20,standard penalty. Probe pending.

Probe metrics: {"passno": 28, "baseline_gap": 0.06999068294422646, "gap": 0.07772850431238365, "ratio": 1.1105550202206662, "clean_drift": 0.002659608923522665, "language_drift": 0.0010448231743790618, "raw_change": 0.017018407894235844, "correction_fraction": -0.0871046707034111, "edited_gap": 0.07747140123896762, "category_ratio": {"Background Textures": 0.9646, "Camera Viewpoints": 0.9151, "Light Conditions": 0.9967, "Objects Layout": 0.9941, "Robot Initial States": 1.402, "Sensor Noise": 0.9882}}

### Pass29 isolated null-carrier gate development
{"passes": [29], "n": 96, "original_success": 47, "edited_success": 47, "wrong_to_right": 2, "right_to_wrong": 2, "diff_pp": 0.0, "se_pp": 2.0942695414584778, "ci95_pp": [-4.104768301258616, 4.104768301258616]}
No net gain. Next controlled test adds the legacy detector-dependent K/V payload in channel1066, orthogonalized against the null carrier. Original clean channel retained; detector-off computation unchanged from pass28/29 apart from roundoff.

### Pass30 null-carrier gate plus legacy K/V payload
{"passes": [30], "n": 96, "original_success": 47, "edited_success": 51, "wrong_to_right": 5, "right_to_wrong": 1, "diff_pp": 4.166666666666666, "se_pp": 2.5290707444453506, "ci95_pp": [-0.7903119924462212, 9.123645325779552]}
Saved immutable candidate_null_payload1. Same detector-off computation as isolated gate; retains original text output channel. Next safety repeats pass27's80 strip+48 language.

### Pass31 null-carrier+payload safety FAILED
{"passes": [31], "n": 128, "original_success": 90, "edited_success": 83, "wrong_to_right": 5, "right_to_wrong": 12, "diff_pp": -5.46875, "se_pp": 3.197215537373303, "ci95_pp": [-11.735292453251674, 0.7977924532516738]}
Clean59->57/80; language31->26/48. Detector disabled for causal safety ablation (only6 text-entry biases changed to-12800); must restore before repair evaluation.

### Pass32 disabled-detector safety ablation
{"passes": [32], "n": 128, "original_success": 91, "edited_success": 86, "wrong_to_right": 2, "right_to_wrong": 7, "diff_pp": -3.90625, "se_pp": 2.3272879876501382, "ci95_pp": [-8.467734455794272, 0.6552344557942709]}
Clean60->58/80,language31->28/48. Most safety harm is structural wiring. ALL gate edits removed. New candidate: video text-entry orthogonal rank6 erasure of the six measured novel tag directions, protected against diagnostic clean/prefix/proprio. Only mot.mixtures.video.text_embedding.0.weight edited.

### Pass33 video-entry novel rank6 projection
{"passes": [33], "n": 96, "original_success": 47, "edited_success": 48, "wrong_to_right": 3, "right_to_wrong": 2, "diff_pp": 1.0416666666666665, "se_pp": 2.339024231752452, "ci95_pp": [-3.5428208275681397, 5.626154160901473]}
Weak +1/96; not validated, snapshot candidate_video_tag_rank6.npy. Removed. Next candidate: null-carrier+legacy payload limited to video blocks0-9. Later blocks keep original queries and receive no detector K/V payload. Carrier projected out before ordinary late-layer K/V reads.

### Pass34 early video gate development
{"passes": [34], "n": 96, "original_success": 47, "edited_success": 50, "wrong_to_right": 4, "right_to_wrong": 1, "diff_pp": 3.125, "se_pp": 2.3194094264961715, "ci95_pp": [-1.421042475932496, 7.671042475932496]}
### Pass35 early video gate development safety
{"passes": [35], "n": 128, "original_success": 89, "edited_success": 88, "wrong_to_right": 5, "right_to_wrong": 6, "diff_pp": -0.78125, "se_pp": 2.600370428544673, "ci95_pp": [-5.8779760399475585, 4.3154760399475585]}
Clean59->58/80,language30->30/48. FREEZE candidate: frozen_early_video_gate (103 tensors), manifest saved. All action weights original. Gate and payload only video blocks0-9. Remaining15 passes:6 raw tagged validation sets (768),5 matched strip sets (640),4 language sets(512). No tuning on validation. Retain if tagged paired95%CI positive and clean/language point losses each <=2pp; otherwise restore original.

### Pass36 frozen early-video gate, tagged validation0
{"passes": [36], "n": 128, "original_success": 69, "edited_success": 63, "wrong_to_right": 3, "right_to_wrong": 9, "diff_pp": -4.6875, "se_pp": 2.6849351430604322, "ci95_pp": [-9.949972880398448, 0.5749728803984473]}
Negative independent result. Candidate remains frozen; continue preplanned validation, do not include development outcomes in final gain estimate.

### Pass37 frozen early-video gate, tagged validation1
{"passes": [37], "n": 128, "original_success": 64, "edited_success": 62, "wrong_to_right": 2, "right_to_wrong": 4, "diff_pp": -1.5625, "se_pp": 1.916173585724926, "ci95_pp": [-5.318200228020855, 2.193200228020855]}
Pooled validation0+1: {"passes": [36, 37], "n": 256, "original_success": 133, "edited_success": 125, "wrong_to_right": 5, "right_to_wrong": 13, "diff_pp": -3.125, "se_pp": 1.648956146338956, "ci95_pp": [-6.356954046824354, 0.10695404682435372]}
REJECT gate for futility; amend previous15-pass plan rather than spend remaining13 on an independently harmful edit. No gate retained. Validation0/1 excluded for replacement.
New final candidate: video text-entry rank6 OBLIQUE full novel-prototype erasure P=I-A*QD^T, QD^T*A=I, QD orthogonal to protected clean/prefix/proprio span. Unlike pass33 orthogonal erasure, annihilates full category-mean novel context vectors. One tensor; no attention or hidden-neuron rewiring. Development38/39 then fresh tag validation2-5(512),strip2-4(384),language0-3(512).

### Pass38 full novel-prototype oblique projection REJECTED
{"passes": [38], "n": 96, "original_success": 47, "edited_success": 27, "wrong_to_right": 0, "right_to_wrong": 20, "diff_pp": -20.833333333333336, "se_pp": 4.166666666666666, "ci95_pp": [-29.0, -12.66666666666667]}
20 losses,0 gains: full prototype components contain indispensable information. Removed immediately; no safety rollout warranted.
FINAL TEST CANDIDATE: frozen_video_orthogonal_rank6, one video text-entry tensor (pass33 development47->48,3 gains/2 losses). Original minus W*Q6*Q6^T; relative weight norm change.0391. No gating, no other edits. Use pass39 development safety,40-43 fresh tag validation2-5,44-46 strip2-4,47-50 language0-3. No further candidate search. Retain only positive tagged95%CI and clean/language losses each<=2pp; otherwise original.

### Pass39 rank6 projection development safety
128 paired strip episodes including48 language:87->87 successes;6 gains/6 losses,
difference0.0pp,SE2.7pp. Clean(non-Language)59->61/80(+2.5pp);
Language30->28/48(-4.17pp), uncertain with this small sample.
Frozen one-tensor candidate unchanged. Independent validation remains:
40-43 tag_validation_2..5 raw;44-46 tag_validation_2..4 strip;
47-50 language_validation_0..3 raw. No further candidate tuning.

### Pass40 frozen rank6 independent tagged: tag_validation_2
{"passes": [40], "n": 128, "original_success": 57, "edited_success": 55, "wrong_to_right": 3, "right_to_wrong": 5, "diff_pp": -1.5625, "se_pp": 2.2140542385542052, "ci95_pp": [-5.902046307566242, 2.777046307566242]}
Pooled tagged: {"passes": [40], "n": 128, "original_success": 57, "edited_success": 55, "wrong_to_right": 3, "right_to_wrong": 5, "diff_pp": -1.5625, "se_pp": 2.2140542385542052, "ci95_pp": [-5.902046307566242, 2.777046307566242]}
No weight changes.

### Pass41 frozen rank6 independent tagged: tag_validation_3
{"passes": [41], "n": 128, "original_success": 56, "edited_success": 57, "wrong_to_right": 1, "right_to_wrong": 0, "diff_pp": 0.78125, "se_pp": 0.78125, "ci95_pp": [-0.75, 2.3125]}
Pooled tagged: {"passes": [40, 41], "n": 256, "original_success": 113, "edited_success": 112, "wrong_to_right": 4, "right_to_wrong": 5, "diff_pp": -0.390625, "se_pp": 1.1739157068134967, "ci95_pp": [-2.6914997853544533, 1.9102497853544533]}
No weight changes.

### Pass42 frozen rank6 independent tagged: tag_validation_4
{"passes": [42], "n": 128, "original_success": 64, "edited_success": 56, "wrong_to_right": 4, "right_to_wrong": 12, "diff_pp": -6.25, "se_pp": 3.087869968907676, "ci95_pp": [-12.302225139059043, -0.19777486094095575]}
Pooled tagged: {"passes": [40, 41, 42], "n": 384, "original_success": 177, "edited_success": 168, "wrong_to_right": 8, "right_to_wrong": 17, "diff_pp": -2.34375, "se_pp": 1.2982700908252898, "ci95_pp": [-4.888359378017568, 0.2008593780175678]}
No weight changes.

### Pass43 frozen rank6 independent tagged: tag_validation_5
{"passes": [43], "n": 128, "original_success": 72, "edited_success": 68, "wrong_to_right": 4, "right_to_wrong": 8, "diff_pp": -3.125, "se_pp": 2.7027754399395, "ci95_pp": [-8.422439862281418, 2.1724398622814194]}
Pooled tagged: {"passes": [40, 41, 42, 43], "n": 512, "original_success": 249, "edited_success": 236, "wrong_to_right": 12, "right_to_wrong": 25, "diff_pp": -2.5390625, "se_pp": 1.1838851193317426, "ci95_pp": [-4.859477333890215, -0.21864766610978448]}
No weight changes.

### Final rank6 efficacy decision after pass43
REJECTED for submission. Independent tagged512:249->236,12 gains/25 losses. Projection remains active ONLY for preplanned clean/paraphrase characterization (passes44-50); restore original afterward. No replacement candidate will be selected from validation outcomes.

### Pass44 frozen rank6 independent clean: tag_validation_2
{"passes": [44], "n": 128, "original_success": 88, "edited_success": 86, "wrong_to_right": 2, "right_to_wrong": 4, "diff_pp": -1.5625, "se_pp": 1.916173585724926, "ci95_pp": [-5.318200228020855, 2.193200228020855]}
Pooled clean: {"passes": [44], "n": 128, "original_success": 88, "edited_success": 86, "wrong_to_right": 2, "right_to_wrong": 4, "diff_pp": -1.5625, "se_pp": 1.916173585724926, "ci95_pp": [-5.318200228020855, 2.193200228020855]}
No weight changes.

### Pass45 frozen rank6 independent clean: tag_validation_3
{"passes": [45], "n": 128, "original_success": 87, "edited_success": 84, "wrong_to_right": 3, "right_to_wrong": 6, "diff_pp": -2.34375, "se_pp": 2.3437499999999996, "ci95_pp": [-6.937499999999999, 2.249999999999999]}
Pooled clean: {"passes": [44, 45], "n": 256, "original_success": 175, "edited_success": 170, "wrong_to_right": 5, "right_to_wrong": 10, "diff_pp": -1.953125, "se_pp": 1.5109051990535214, "ci95_pp": [-4.914499190144902, 1.0082491901449018]}
No weight changes.

### Pass46 frozen rank6 independent clean: tag_validation_4
{"passes": [46], "n": 128, "original_success": 82, "edited_success": 84, "wrong_to_right": 6, "right_to_wrong": 4, "diff_pp": 1.5625, "se_pp": 2.476358441722394, "ci95_pp": [-3.291162545775892, 6.416162545775892]}
Pooled clean: {"passes": [44, 45, 46], "n": 384, "original_success": 257, "edited_success": 254, "wrong_to_right": 11, "right_to_wrong": 14, "diff_pp": -0.78125, "se_pp": 1.3031707817603402, "ci95_pp": [-3.3354647322502666, 1.7729647322502666]}
No weight changes.

### Pass47 frozen rank6 independent paraphrase: language_validation_0
{"passes": [47], "n": 128, "original_success": 84, "edited_success": 85, "wrong_to_right": 8, "right_to_wrong": 7, "diff_pp": 0.78125, "se_pp": 3.03686619509478, "ci95_pp": [-5.171007742385769, 6.733507742385769]}
Pooled paraphrase: {"passes": [47], "n": 128, "original_success": 84, "edited_success": 85, "wrong_to_right": 8, "right_to_wrong": 7, "diff_pp": 0.78125, "se_pp": 3.03686619509478, "ci95_pp": [-5.171007742385769, 6.733507742385769]}
No weight changes.

### Pass48 frozen rank6 independent paraphrase: language_validation_1
{"passes": [48], "n": 128, "original_success": 98, "edited_success": 96, "wrong_to_right": 5, "right_to_wrong": 7, "diff_pp": -1.5625, "se_pp": 2.713423316130324, "ci95_pp": [-6.880809699615434, 3.755809699615434]}
Pooled paraphrase: {"passes": [47, 48], "n": 256, "original_success": 182, "edited_success": 181, "wrong_to_right": 13, "right_to_wrong": 14, "diff_pp": -0.390625, "se_pp": 2.03357592134641, "ci95_pp": [-4.376433805838964, 3.595183805838964]}
No weight changes.

### Pass49 frozen rank6 independent paraphrase: language_validation_2
{"passes": [49], "n": 128, "original_success": 92, "edited_success": 95, "wrong_to_right": 8, "right_to_wrong": 5, "diff_pp": 2.34375, "se_pp": 2.8202471494108172, "ci95_pp": [-3.1839344128452014, 7.871434412845201]}
Pooled paraphrase: {"passes": [47, 48, 49], "n": 384, "original_success": 274, "edited_success": 276, "wrong_to_right": 21, "right_to_wrong": 19, "diff_pp": 0.5208333333333333, "se_pp": 1.6489536197122974, "ci95_pp": [-2.7111157613027697, 3.7527824279694357]}
No weight changes.

### Pass50 frozen rank6 independent paraphrase: language_validation_3
{"passes": [50], "n": 128, "original_success": 88, "edited_success": 89, "wrong_to_right": 7, "right_to_wrong": 6, "diff_pp": 0.78125, "se_pp": 2.8270552393399617, "ci95_pp": [-4.759778269106325, 6.322278269106325]}
Pooled paraphrase: {"passes": [47, 48, 49, 50], "n": 512, "original_success": 362, "edited_success": 365, "wrong_to_right": 28, "right_to_wrong": 25, "diff_pp": 0.5859375, "se_pp": 1.4230510256546045, "ci95_pp": [-2.2032425102830246, 3.3751175102830246]}
No weight changes.

## FINAL SUBMISSION AFTER PASS50
ALL EDITS REMOVED; original checkpoint submitted. Verified edited()==[] and former edited tensor equals load_original exactly.
Frozen rank6 independent results:
{
  "tagged": {
    "passes": [
      40,
      41,
      42,
      43
    ],
    "n": 512,
    "original_success": 249,
    "edited_success": 236,
    "wrong_to_right": 12,
    "right_to_wrong": 25,
    "diff_pp": -2.5390625,
    "se_pp": 1.1838851193317426,
    "ci95_pp": [
      -4.859477333890215,
      -0.21864766610978448
    ]
  },
  "clean": {
    "passes": [
      44,
      45,
      46
    ],
    "n": 384,
    "original_success": 257,
    "edited_success": 254,
    "wrong_to_right": 11,
    "right_to_wrong": 14,
    "diff_pp": -0.78125,
    "se_pp": 1.3031707817603402,
    "ci95_pp": [
      -3.3354647322502666,
      1.7729647322502666
    ]
  },
  "paraphrase": {
    "passes": [
      47,
      48,
      49,
      50
    ],
    "n": 512,
    "original_success": 362,
    "edited_success": 365,
    "wrong_to_right": 28,
    "right_to_wrong": 25,
    "diff_pp": 0.5859375,
    "se_pp": 1.4230510256546045,
    "ci95_pp": [
      -2.2032425102830246,
      3.3751175102830246
    ]
  }
}
Matched suffix penalty:
{
  "original": {
    "n": 384,
    "raw_success": 177,
    "clean_success": 257,
    "clean_minus_raw_pp": 20.833333333333336,
    "paired_se_pp": 2.492009277829658
  },
  "edited": {
    "n": 384,
    "raw_success": 168,
    "clean_success": 254,
    "clean_minus_raw_pp": 22.395833333333336,
    "paired_se_pp": 2.371874279150996
  }
}
No candidate achieved a validated tagged gain while satisfying safety. The requested robust repair was not achieved. Full final summary: final_report.json.
