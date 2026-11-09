# FastWAM second session

Read prior_notes.md. Earlier linear/head edits failed; selective context gate gave small tagged gain and language loss. No prior edits applied. Goal: diagnose token-level contamination and separate video/action routes; reserve independent validation.
Constraints: weight edits only, no local network execution or training. 0/50 forward passes used.

Fixed hash/category/suite-balanced disjoint splits in splits.json: diag24, language_diag32, development96, tagged validation A-D128 each, language A-C128 each. Diagnostic inputs will be raw/strip matched.

Pass1-2: original raw12/24 vs strip13/24. Diagnostic set has camera0/4 in both; small/low-power gap. Saved first-observation pairs, attention episode means for both experts. Development split remains separate from validation.

Pass3-4: development raw49/96 (51.04%), strip65/96 (67.71%); diagnostic paraphrases26/32. Combined120 raw/clean pairs copied unchanged into replays; 32 language inputs. All image/proprio/seed pairs checked identical.

Correction: 116/120 pairs have exactly identical images and proprio; strict tag-only action metrics must exclude indices [22, 38, 59, 115]. Copied replay paths refused without charge; replaced by symlinks to registered original files.

Pass5: original common-noise reference on272 inputs saved. Next diagnostic: video text-entry projection onto exact9D proprio weight+bias span (one tensor); removes text-specific input. This is not a selected repair.

Probe metrics: {"pass": 6, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.19381114117392237, "gap_ratio": 3.144933778102811, "edited_raw_clean_gap": 0.024998909376050815, "clean_drift": 0.1862382786064431, "language_drift": 0.12036455953171618, "raw_change": 0.16451271156495548, "correction_fraction": -0.8821513652801514}

Pass6 intervention was video-only full text-specific ablation, proprio span preserved. Now action-only identical ablation for pass7; video restored to original.

Probe metrics: {"pass": 7, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.08692082114362157, "gap_ratio": 1.4104464004455777, "edited_raw_clean_gap": 0.05476892068494885, "clean_drift": 0.05406318830881489, "language_drift": 0.04517315368767735, "raw_change": 0.04355415646991809, "correction_fraction": -0.24493569135665894}

Causal localization: video text ablation shrank raw-clean gap .06163->.02500 but clean drift.1862; action text ablation left gap.05477, clean drift.05406. Neither retained.
TEMPORARY INSTRUMENTATION for passes8-10: transplant disjoint1023-feature slices of original video text GELU into action text MLP; three controlled cross-attention layers read uniform / prior-detector-masked / detector-emphasized means. Original video MLP features3069/3072; exclude3 lowest weight-energy features. Not a candidate, remove all diagnostic edits after acquisition.

Pass8-9 readouts validated: temporal RMS below1e-6; masking leaves measured clean/paraphrase means exactly unchanged. Slice0 uniform tag contrast RMS.00633, masked.00439: conservative suffix masking leaves substantial context contamination. Final slice pending; instrumentation is still active and must be cleared.

Pass10 complete. All48 diagnostic tensors cleared. Readout data saved video_context_features.pt (128 inputs x uniform/masked/emphasized x3072 hidden channels;3 lowest-energy channels omitted). Candidate construction must use static linear algebra only, not loss fitting.

Candidate video_hidden_P1: one tensor video.text_embedding.2.weight right-composed with rank1 oblique projector. Uniform post-GELU contrast top mode73% energy; masked top mode47%. Protect79D measured clean/paraphrase means plus linear input-proprio/bias span; static SVD, no fitting. Must evaluate clean token effects empirically.

Probe metrics: {"pass": 11, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.06003225405373173, "gap_ratio": 0.9741311175697951, "edited_raw_clean_gap": 0.060006564740404375, "clean_drift": 0.0011905509695017934, "language_drift": 0.001239090640640736, "raw_change": 0.01932656424510607, "correction_fraction": 0.07470933347940445}

Probe metrics: {"pass": 12, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.05907658149623656, "gap_ratio": 0.9586236142928015, "edited_raw_clean_gap": 0.061999419088984244, "clean_drift": 0.018596861180178736, "language_drift": 0.0016679545100961913, "raw_change": 0.019543068935665202, "correction_fraction": 0.09080337733030319}

Next candidate video_masked_P48: full48D observed masked-prefix contrast span, same protected clean/proprio directions. Tests whether the earlier small-rank methods missed distributed tag contamination. One tensor, video text MLP output.

Probe metrics: {"pass": 13, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.05294705210482181, "gap_ratio": 0.8591609935674253, "edited_raw_clean_gap": 0.05166380631751359, "clean_drift": 0.003126988199161972, "language_drift": 0.003684277278148568, "raw_change": 0.02766603494044892, "correction_fraction": 0.23169076442718506}

Next key-only causal candidate: video cross_attn.k weights/biases in all30 blocks right-projected away from8 uniform context-contrast directions, protected output-space clean/proprio/bias span. Values, text MLP and action expert original.

Probe metrics: {"pass": 14, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.06700836893495356, "gap_ratio": 1.0873311080192203, "edited_raw_clean_gap": 0.06707615560743464, "clean_drift": 0.0012607198181753812, "language_drift": 0.0013975878058074829, "raw_change": 0.02024121093677415, "correction_fraction": -0.03720477223396301}

Probe metrics: {"pass": 15, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.05734906461061791, "gap_ratio": 0.930591550847338, "edited_raw_clean_gap": 0.057186007734585716, "clean_drift": 0.0013163180840452778, "language_drift": 0.0015091515852457098, "raw_change": 0.019331009813504614, "correction_fraction": 0.11619734764099121}

Candidate video_union: union of all observed unweighted and masked feature contrasts plus high-confidence tag-emphasized feature directions relative to padding. SVD and principal-angle filtering only; no optimization. Rank104, protected clean/proprio/bias/padding span. One video text output tensor.

Probe metrics: {"pass": 16, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.050404122761032524, "gap_ratio": 0.8178973988113576, "edited_raw_clean_gap": 0.0476030182875327, "clean_drift": 0.0052892039533556446, "language_drift": 0.005666096067834513, "raw_change": 0.03400163345422246, "correction_fraction": 0.31772908568382263}

Pass16 union projection: gap ratio.8179, arm-first10 .9414, gripper .5008; clean/lang drift.00529/.00567. Saved candidate_video_union_full. Pass17 tests same fixed rank104 directions at half dose.

Probe metrics: {"pass": 17, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.05652209188750858, "gap_ratio": 0.9171724334801774, "edited_raw_clean_gap": 0.055683010934889685, "clean_drift": 0.002384016708781759, "language_drift": 0.002804298057033065, "raw_change": 0.01946344964544439, "correction_fraction": 0.1292714923620224}

TEMPORARY weighted-context instrumentation active. weighted_readout.py holds installation/extraction. 13 context weightings (uniform and +/-4,+/-16 along three fixed input directions),1018 video GELU features per slice,3 slices. Directions: old tag detector and two clean-context PCs mapped through original W0; exact linear keys via GELU(x)-GELU(-x). All diagnostic edits MUST be cleared after acquisition. No fitting/training.

Passes18-20 video weighted readouts complete: weighted_video_features.pt. Pass21 action weighted readout complete: weighted_action_features.pt; swapped request positions42/43 corrected using registered input paths. All175 diagnostic edits cleared. Projector factors saved weighted_{video,action}_projector.pt. Based only on static SVD/principal angles, not action-target fitting.

Candidate weighted_video_full: one video.text_embedding.2.weight, contrast top128 modes ->118 principal-angle-retained directions, clean protection769D, min-angle.15, dose1. Relative weight change.5095. Probe22 pending.

Probe metrics: {"pass": 22, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.048190448943811415, "gap_ratio": 0.7819765661940364, "edited_raw_clean_gap": 0.04749530898284968, "clean_drift": 0.0023683156520915684, "language_drift": 0.004012676106347218, "raw_change": 0.035096347034269115, "correction_fraction": 0.35642215609550476}

Pass22 weighted video-only: gap ratio.7820; arm first10 .8946; clean drift.00237, language.00401;33/48 arm examples improve. Candidate saved. Added action weighted projector at dose.5 (53 directions,626D protection) for isolated incremental test23.

Probe metrics: {"pass": 23, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.047257334650958396, "gap_ratio": 0.7668351112670954, "edited_raw_clean_gap": 0.045988156987098856, "clean_drift": 0.003119235415928299, "language_drift": 0.004599219045564942, "raw_change": 0.035552611552432424, "correction_fraction": 0.37239164113998413}

Pass23 video1+action.5: gap ratio.7668 vs video-only.7820; arm first10.8839 vs.8946, drift clean.00312/lang.00460. Small action increment. Next pass24 tests video-only expanded contrast rank256 ->226 retained, same769D protected span, relative change.7296.

Probe metrics: {"pass": 24, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.053528398228537846, "gap_ratio": 0.8685943783056277, "edited_raw_clean_gap": 0.05570246029737116, "clean_drift": 0.01948182531689479, "language_drift": 0.004567767761604531, "raw_change": 0.03120015713693379, "correction_fraction": 0.25093093514442444}

Pass24 rank256 video rejected: gap ratio.8686, clean drift.01948 (camera gripper outlier); worse than rank128. Restored candidate_weighted_video1_actionhalf (2 tensors) for paired development96 + diagnostic language32 rollout25. Independent validation splits remain unused.

Pass25 development result for video1/action.5: tagged49->50/96 (+1.04pp); language26->24/32 (-6.25pp, low power). Not sufficient to retain. FINAL diagnostic campaign: masked weighted prefix summaries, using same safe threshold.5 suffix detector to suppress detected suffix tokens in all13 queries. 3 slices1016 video features; masked_readout.py and masked_readout_spec.pt. Temporary instrumentation active, must clear after passes26-28. Plan repair probe29/development30 then freeze and independent validation.

Correction: saved original readout detector was threshold1.0 (norm32,bias-32). New masked readout explicitly uses threshold.5,bias-16. Added block13 uniform unmasked readout to measure detector activity on clean/paraphrases. Request layers0..13.

Pass26 detector verification: mean raw.1330 (min.000122,max.2984), clean mean-1.3e-6/max6.1e-5, language mean1e-6/max3.05e-5. Masked contrast slice norm.1134 vs unmasked.1864. Pass27 second slice saved; final slice pending. All188 diagnostic tensors still active.

Pass28 masked video readouts assembled into masked_video_features.pt; all188 instrumentation edits cleared. Post-restart analysis begins with no active edits.

Candidate masked_video128: masked-prefix contrast rank128, retained114, protected507D clean/paraphrase plus unread/proprio directions. One video text_embedding.2.weight tensor; no attention gating yet. Static SVD, no fitting. Probe29 planned.

Before probe29, strengthened protection to the previous weighted-video protected span plus unread channels: 775D, 112 retained masked contrast directions. No results used for this adjustment.

Probe metrics: {"pass": 29, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.09893061426646434, "gap_ratio": 1.6053268589748475, "edited_raw_clean_gap": 0.09724202152629204, "clean_drift": 0.0038290903749451525, "language_drift": 0.005410165062870439, "raw_change": 0.07227280356599848, "correction_fraction": -0.10085882246494293}

Pass29 masked-prefix projector alone rejected: gap ratio1.6053, arm-first10 1.4439, clean drift.00383/lang.00541. Hypothesis: projector damages still-visible suffix values. Pass30 tests SAME projector plus video-only suffix gate threshold.5/strength32/key64; action untouched. gate_install.py reproduces gate. This is not yet a retained candidate.

Probe metrics: {"pass": 30, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.07792651095009029, "gap_ratio": 1.2644975671275425, "edited_raw_clean_gap": 0.07608223974180046, "clean_drift": 0.004371052515530738, "language_drift": 0.005409730376130118, "raw_change": 0.0479552719881013, "correction_fraction": 0.0032897531054913998}

Pass30 gate+masked projection rejected: gap ratio1.2645, arm-first10 1.1733, clean/lang drift.00437/.00541. All candidate edits cleared. TEMPORARY negative-GELU readout active: negative_readout.py, 3 slices planned31-33. Difference from matching positive masked readouts gives exact weighted pre-GELU linear context summaries (GELU(x)-GELU(-x)=x). All diagnostic edits must be cleared afterward. No training/local network execution.

Pass31 negative video slice0 saved, temporally stable. Temporary instrumentation advanced to slice1.

Pass32 negative video slice1 saved, temporally stable. Temporary instrumentation advanced to final slice2 for pass33.

Pass33 negative readouts complete; all188 diagnostic tensors cleared. linear_video_features.pt and linear_context_coordinates.pt saved. Observable UMT5 row-space coordinates reconstructed algebraically with SVD of original video text-entry matrix; no local network evaluation or fitting.

Candidate linear_context_video: one video text_embedding.0.weight; original bias and all later layers untouched. Broad/moderate queries[0, 2, 3, 6, 7, 10, 11, 13], 99% contrast rank175, retained160, protects864 clean/proprio directions. Clean uncentered energy .999999. Static observable-context SVD projector; probe34 planned.

Probe metrics: {"pass": 34, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.08464599898047129, "gap_ratio": 1.3735333261159228, "edited_raw_clean_gap": 0.08222058779662199, "clean_drift": 0.005034002314950593, "language_drift": 0.012369156260626442, "raw_change": 0.054021777802209604, "correction_fraction": -0.05908278375864029}

Pass34 pre-GELU candidate rejected: gap1.3735, arm10 1.3052, clean/lang drift.00503/.01237. All edits cleared. Pass35 candidate: video-only cross_attn.v.weight in all30 blocks. Output-space image of previously successful weighted118D hidden contrast span, protected clean/proprio/bias span, retained115; values only, keys/text/action original. Static oblique projector, no fitting.

Probe metrics: {"pass": 35, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.04777213906964177, "gap_ratio": 0.7751887373570477, "edited_raw_clean_gap": 0.05119622918832735, "clean_drift": 0.019329352027242442, "language_drift": 0.003402244690247115, "raw_change": 0.03239463564697989, "correction_fraction": 0.33770105242729187}

Pass36 matched key-only test: same output-space projector as pass35 moved from video cross_attn.v.weight to cross_attn.k.weight, all30 layers. Values/text/action original. No subspace or dose change.

Probe metrics: {"pass": 36, "baseline_gap": 0.06162646174726115, "raw_to_clean": 0.05898261524622319, "gap_ratio": 0.9570988431579805, "edited_raw_clean_gap": 0.0587707705484742, "clean_drift": 0.0019935299868434715, "language_drift": 0.0031813455674193467, "raw_change": 0.020385394578949743, "correction_fraction": 0.0966918021440506}

Key-only35/36 comparison: value-only gap.7752, arm10.9054; key-only gap.9571, arm10.9987. Stronger evidence that video text values carry the useful correction. Restored previously saved one-tensor weighted_video_full (probe22) for development rollout37, without action edit. Request is same development96 + diagnostic language32 as25. Reserved validation sets remain untouched.

Pass37 development video text-output only: tagged49->49/96 (0pp), language26->25/32 (-3.125pp). Not retained on development evidence. Restored pass35 value-only projector (30 tensors) for same development rollout38.

Pass38 value-only full development: tagged49->49/96 (0pp), language26->27/32 (+3.125pp). Pass39 same directions/layers at half dose, final development test. No validation variants used for choice. Remaining11 passes reserved for frozen-candidate independent raw/strip/language validation.

FROZEN FINAL CANDIDATE after pass39: video cross_attn.v.weight only, all30 blocks, rank115 oblique context-contrast correction at dose.5. All text embeddings, keys/queries, action expert, self-attention, MLPs and biases original. Copy frozen_video_value_half; frozen_manifest.json. No tuning on validation. Plan40-43 raw A-D512,44-47 strip A-D512,48-50 Language A-C384. Retain only if independent tagged gain clearly exceeds paired SE (95% CI lower >0), and clean/language point losses <=2pp; otherwise revert all edits.

Frozen validation raw A/pass40: {"passes": [40], "n": 128, "original": 64, "edited": 64, "wrong_to_right": 3, "right_to_wrong": 3, "diff_pp": 0.0, "se_pp": 1.9211831984195997, "ci95_pp": [-3.7655190689024156, 3.7655190689024156]}. No weight changes.

Frozen validation raw B/pass41: {"passes": [41], "n": 128, "original": 52, "edited": 50, "wrong_to_right": 4, "right_to_wrong": 6, "diff_pp": -1.5625, "se_pp": 2.476358441722394, "ci95_pp": [-6.416162545775892, 3.291162545775892]}. Pooled A+B: {"passes": [40, 41], "n": 256, "original": 116, "edited": 114, "wrong_to_right": 7, "right_to_wrong": 9, "diff_pp": -0.78125, "se_pp": 1.5647961070432297, "ci95_pp": [-3.84825036980473, 2.28575036980473]}. No weight changes; no validated gain so far.

Frozen validation raw C/pass42: {"passes": [42], "n": 128, "original": 56, "edited": 57, "wrong_to_right": 4, "right_to_wrong": 3, "diff_pp": 0.78125, "se_pp": 2.073956712274816, "ci95_pp": [-3.283705156058639, 4.846205156058639]}. Pooled A-C: {"passes": [40, 41, 42], "n": 384, "original": 172, "edited": 171, "wrong_to_right": 11, "right_to_wrong": 12, "diff_pp": -0.26041666666666663, "se_pp": 1.2504730347699742, "ci95_pp": [-2.711343814815816, 2.190510481482483]}. Frozen tensor hashes verified after restart; no weight changes.

Frozen validation raw D/pass43: {"passes": [43], "n": 128, "original": 57, "edited": 59, "wrong_to_right": 6, "right_to_wrong": 4, "diff_pp": 1.5625, "se_pp": 2.476358441722394, "ci95_pp": [-3.291162545775892, 6.416162545775892]}. Final raw A-D: {"passes": [40, 41, 42, 43], "n": 512, "original": 229, "edited": 230, "wrong_to_right": 17, "right_to_wrong": 16, "diff_pp": 0.1953125, "se_pp": 1.1230489522602674, "ci95_pp": [-2.005863446430124, 2.396488446430124]}. Predeclared tagged gain criterion FAILED. Candidate will be reverted after completing planned strip/language checks; no tuning or replacement chosen from validation results.

Frozen validation strip_A/pass44: {"passes": [44], "n": 128, "original": 95, "edited": 92, "wrong_to_right": 0, "right_to_wrong": 3, "diff_pp": -2.34375, "se_pp": 1.3424675715302161, "ci95_pp": [-4.974986440199224, 0.28748644019922365]}. Pooled: {"passes": [44], "n": 128, "original": 95, "edited": 92, "wrong_to_right": 0, "right_to_wrong": 3, "diff_pp": -2.34375, "se_pp": 1.3424675715302161, "ci95_pp": [-4.974986440199224, 0.28748644019922365]}. No weight changes.

Frozen validation strip_B/pass45: {"passes": [45], "n": 128, "original": 80, "edited": 83, "wrong_to_right": 7, "right_to_wrong": 4, "diff_pp": 2.34375, "se_pp": 2.5929672236898638, "ci95_pp": [-2.738465758432133, 7.425965758432133]}. Pooled: {"passes": [44, 45], "n": 256, "original": 175, "edited": 175, "wrong_to_right": 7, "right_to_wrong": 7, "diff_pp": 0.0, "se_pp": 1.4644479653225153, "ci95_pp": [-2.87031801203213, 2.87031801203213]}. No weight changes.

Frozen validation strip_C/pass46: {"passes": [46], "n": 128, "original": 78, "edited": 85, "wrong_to_right": 9, "right_to_wrong": 2, "diff_pp": 5.46875, "se_pp": 2.555629533115372, "ci95_pp": [0.45971611509387067, 10.47778388490613]}. Pooled: {"passes": [44, 45, 46], "n": 384, "original": 253, "edited": 260, "wrong_to_right": 16, "right_to_wrong": 9, "diff_pp": 1.8229166666666667, "se_pp": 1.3004504556923069, "ci95_pp": [-0.7259662264902549, 4.371799559823589]}. No weight changes.

Frozen validation strip_D/pass47: {"passes": [47], "n": 128, "original": 90, "edited": 86, "wrong_to_right": 4, "right_to_wrong": 8, "diff_pp": -3.125, "se_pp": 2.7027754399395, "ci95_pp": [-8.422439862281418, 2.1724398622814194]}. Pooled: {"passes": [44, 45, 46, 47], "n": 512, "original": 343, "edited": 346, "wrong_to_right": 20, "right_to_wrong": 17, "diff_pp": 0.5859375, "se_pp": 1.1889189347294897, "ci95_pp": [-1.7443436120698, 2.9162186120698]}. No weight changes.

Final clean A-D complete. Tag penalty (strip minus raw) pp: {"original": 22.265625, "edited": 22.65625}. No closed-loop tag-gap closure demonstrated. Remaining passes48-50 are untouched Language A-C; candidate remains frozen.

Frozen validation language_A/pass48: {"passes": [48], "n": 128, "original": 94, "edited": 97, "wrong_to_right": 5, "right_to_wrong": 2, "diff_pp": 2.34375, "se_pp": 2.0646668258753222, "ci95_pp": [-1.7029969787156318, 6.390496978715632]}. Pooled: {"passes": [48], "n": 128, "original": 94, "edited": 97, "wrong_to_right": 5, "right_to_wrong": 2, "diff_pp": 2.34375, "se_pp": 2.0646668258753222, "ci95_pp": [-1.7029969787156318, 6.390496978715632]}. No weight changes.

Frozen validation language_B/pass49: {"passes": [49], "n": 128, "original": 86, "edited": 86, "wrong_to_right": 3, "right_to_wrong": 3, "diff_pp": 0.0, "se_pp": 1.9211831984195997, "ci95_pp": [-3.7655190689024156, 3.7655190689024156]}. Pooled: {"passes": [48, 49], "n": 256, "original": 180, "edited": 183, "wrong_to_right": 8, "right_to_wrong": 5, "diff_pp": 1.171875, "se_pp": 1.4092679353707782, "ci95_pp": [-1.590290153326725, 3.934040153326725]}. No weight changes.
