# FastWAM tag-invariance weight edit

## Constraints
Direct weight analysis/edits only; no training or local model execution. Use only listed practice variants; preserve clean and paraphrased language.

## Plan
1. Inspect architecture/weights and establish matched raw/strip diagnostics.
2. Localize tag influence in action/video cross-attention, test causal doses.
3. Validate a chosen edit on reserved practice variants with paired original/edited rollouts; test strip and Language Instructions separately.

## Pass log
No forward passes used yet.

## Current edit
None.

## Fixed splits
Saved in splits.json: diagnostic=24, development=96, validation_a/b=128 each (tagged categories), language_a/b=128 each. All disjoint, selected deterministically by hashes and category/suite balancing.

### Passes 1-2: original diagnostic raw vs strip
24 matched episodes, raw 13/24 (54.17%), strip 20/24 (83.33%). By category raw->strip: Background 2->3/4; Camera 0->2/4; Light 4->4/4; Layout 2->3/4; Initial States 3->4/4; Noise 2->4/4. All-layer attn_out and replay inputs saved.

### Pass 3: attention-map probe
Four initial inputs (raw/strip pairs from goal Camera and Noise), layers 0,10,12,23. Tool exposes action-to-video maps [24,32,98], not cross-text attention. Cannot directly attribute attention mass to suffix tokens with this interface. Will localize causally by cross-attention weight interventions.

### Pass 4 details / API restriction
Action-only controlled attention outputs were returned by probe; video outputs are available from rollout episode means. Uniform whole-tensor output scaling was rejected by save_edit, no edits made. Moving to targeted low-rank contrast-subspace interventions, protecting the observed clean-output span. No local model execution or training.

### Candidate action_self_P4
For action layers 1-29, left-project self_attn.o weight and bias with (I-UU^T), U=top 4 tagged-clean difference directions after removing the span of 24 clean mean outputs (from probe 4). Static SVD/projection, no training. Relative changed-parameter norm=0.090744. Probe pending.

### Pass 5: action_self_P4 controlled probe
{"pass": 5, "raw_to_original_clean": 0.07932981012500152, "clean_drift": 0.02842921720473182, "edited_raw_clean_gap": 0.07342959402848001, "raw_change": 0.02890110272961311, "original_raw_clean_gap": 0.07557863531824152, "gap_ratio": 1.0496327406675816, "delta_alignment": 0.0581844188272953}
Next: apply identical output-space directions to action cross_attn.o, not self_attn.o, to compare injection vs propagation.

### Pass 6: action_cross_P4 controlled probe
{"pass": 6, "raw_to_original_clean": 0.0755015976323416, "clean_drift": 0.0030451581014146276, "edited_raw_clean_gap": 0.07548024200114467, "raw_change": 0.002216683153548651, "original_raw_clean_gap": 0.07557863531824152, "gap_ratio": 0.9989806949334883, "delta_alignment": 0.04940061643719673}

### Candidate video_cross_H6
Zero 6/24 video cross-attention output-head column groups in each layer; ranked by weight energy writing clean-protected contrast directions divided by energy writing clean directions. Biases untouched. This is a targeted-head causal screen, not evidence that those heads directly attend to tags. No self-attention or action weights edited.

### Pass 7: video_cross_H6 controlled probe
{"pass": 7, "raw_to_original_clean": 0.08088009061662182, "clean_drift": 0.040231865646138044, "edited_raw_clean_gap": 0.06911937256264532, "raw_change": 0.0289688264595742, "original_raw_clean_gap": 0.07557863531824152, "gap_ratio": 1.0701448931441708, "delta_alignment": 0.0022229491733014584}
Next: matched 6/24-head intervention in action cross-attention, video unchanged.

### Pass 8: action_cross_H6 controlled probe
{"pass": 8, "raw_to_original_clean": 0.07993553027072069, "clean_drift": 0.021790985412534384, "edited_raw_clean_gap": 0.07303904951468992, "raw_change": 0.018351915114514228, "original_raw_clean_gap": 0.07557863531824152, "gap_ratio": 1.0576471767998117, "delta_alignment": -0.1228417456150055}
All trial edits cleared after this comparison. No candidate yet validated by rollout.

### Pass 9: original residual reference
48 raw/strip inputs, layers 0,1,3,7,12,18,23,29, per-denoising-step means.
Next candidate action_head_P8: right-project action head weight away from 8 leading tag contrast directions orthogonal to the span of 240 clean residual means. Only one tensor changed; static orthogonal erasure, no action-target fitting.

### Pass 10: action_head_P8 controlled probe
{"pass": 10, "raw_to_original_clean": 0.0746155006459284, "clean_drift": 0.006325739385073019, "edited_raw_clean_gap": 0.07372260139582645, "raw_change": 0.006496561772056333, "original_raw_clean_gap": 0.07557863531824152, "gap_ratio": 0.98725652205471, "delta_alignment": 0.19028720259666443}

### Candidate text_outliers_2_half
No text contexts found in saved artifacts. Action/video text projections share corresponding high-norm rows (action 571,934; video 1714,2804). Attenuate text-only components of those four rows by 0.5, exactly retaining components along proprio_encoder weight+bias span. Hypothesis: shared outlier nonlinear features amplify tag perturbations. Only two tensors edited; all earlier candidates cleared.

### Pass 11: text_outliers_2_half controlled probe
{"pass": 11, "raw_to_original_clean": 0.07522528111837991, "clean_drift": 0.004178204111960896, "edited_raw_clean_gap": 0.07554542327785005, "raw_change": 0.0044314715553247916, "original_raw_clean_gap": 0.07557863531824152, "gap_ratio": 0.9953246813947654, "delta_alignment": 0.10886801034212112}

## Next diagnostic: direct text-context mean readout
Previous candidates did not materially restore clean actions; no repair selected. Plan temporary weight-only instrumentation in action block 0: uniform-query attention reads selected context coordinates, with other writes to those residual coordinates disabled. GELU(x)-GELU(-x)=x permits an exact linear readout through the existing text MLP, without editing frozen UMT5 or running any network locally. All instrumentation will be removed before repair evaluation. Collect 32 disjoint language-diagnostic paraphrases first; validation language sets remain unused.

### Pass 12: diagnostic paraphrases
Original 17/32 success (53.1%), saved initial inputs. 32 variants disjoint from all reserved validation sets.

### CURRENT: TEMPORARY CONTEXT READOUT, NOT A REPAIR
16 action tensors instrumented in text MLP, input rows and block 0. Readout context dimensions 0:512; must clear after acquisition. 80 replay inputs: 24 raw/clean pairs + 32 language diagnostic examples.

### Pass 13: context dimensions 0:512 acquired
Stored context_slice_0000.npy; temporary instrumentation advances to 512:1024.

### Pass 14: context dimensions 512:1024 acquired
Stored context_slice_0512.npy; temporary readout now 1024:1536.

### Pass 15: context dimensions 1024:1536 acquired
Saved context_slice_1024.npy; readout is time-invariant and input order matches passes 13-14. Temporary instrumentation now reads 1536:2048. Still no repair selected.

### Pass 16: context dimensions 1536:2048 acquired
Temporary readout advances to 2048:2560. Probe layers must be integer indices.

### Pass 17: context dimensions 2048:2560 acquired
Temporary readout advances to 2560:3072.

### Pass 18: context dimensions 2560:3072 acquired
Temporary readout advances to 3072:3584.

### Pass 19: context dimensions 3072:3584 acquired
Temporary readout advances to final slice 3584:4096. After pass 20, remove ALL 16 instrumentation edits before applying a repair.

### Pass 20: full text-context acquisition complete
Saved context_means_80x4096.npy and text_tag_basis.pt. ALL temporary instrumentation removed. Tag basis is SVD of paired context-mean differences after orthogonal removal of 24 clean means, 32 paraphrase means, and the 9D proprio encoder weight+bias span. No optimization/training.

### Pass 21: matched original reference
80-input reference includes all raw/clean pairs and 32 paraphrases, noise_seed=0. Candidate text-entry rank-4 full projection applied to action/video text_embedding.0.weight only; biases and all other tensors original. Protects measured clean/paraphrase means and exact proprio span, not necessarily every clean token.

### Pass 22: both-expert text-entry rank-4 projection
{"pass": 22, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.07587848267055199, "gap_ratio": 1.004713622601619, "edited_raw_clean_gap": 0.07577362909301187, "clean_drift": 0.003160395170565966, "language_drift": 0.00253749689147854, "raw_change": 0.004678472943169201, "correction_alignment": -0.045295193791389465, "fraction_corrected": -0.002805950352922082, "by_category": {"Background Textures": {"gap_ratio": 1.0232194671996242, "clean_drift": 0.0026903127594067643}, "Camera Viewpoints": {"gap_ratio": 1.0102482059494302, "clean_drift": 0.0022734253844677826}, "Light Conditions": {"gap_ratio": 1.0116253487034612, "clean_drift": 0.0015797325648290897}, "Objects Layout": {"gap_ratio": 1.0149715124060874, "clean_drift": 0.005907154922690813}, "Robot Initial States": {"gap_ratio": 1.0016673286050632, "clean_drift": 0.00246452598715535}, "Sensor Noise": {"gap_ratio": 0.9990473344813566, "clean_drift": 0.0020145514349936364}}}

### Next: thresholded context detector diagnostic
Rank-4 text erasure discarded: gap ratio 1.0047, no improvement. Temporary 512-feature readout measures GELU(g*(u.x-t))/g across 8 directions, 32 thresholds and gains 8/32. Goal: test whether tag-specific attention gating is feasible. All diagnostic edits must be removed before gate candidate.

### Pass 23: selective detector found
Protected mean contrast, threshold .5, GELU gain32: mean response raw=.1304, minimum raw=.00247; all 56 clean/paraphrase responses effectively zero. Instrumentation removed. New action-only attention gate: one low-cost text hidden neuron/output channel, one low-weight-energy q/k coordinate per head, constant query and negative detector key, q norm gain strength6, key gain64. Video, self-attn, MLPs and action head unchanged. Gate detector exactly orthogonal to proprio span. Candidate not yet evaluated.

### Pass 24: action-only context gate strength6
{"pass": 24, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.07513456989015123, "gap_ratio": 0.9948634084408902, "edited_raw_clean_gap": 0.07503351466755898, "clean_drift": 0.00115045573634545, "language_drift": 0.0011148006862259849, "raw_change": 0.0093111192248976, "correction_alignment": 0.10320055484771729, "fraction_corrected": 0.012723530642688274, "by_category": {"Background Textures": {"gap_ratio": 0.9483697377259845, "clean_drift": 0.0012226948497503938}, "Camera Viewpoints": {"gap_ratio": 0.9931323542973629, "clean_drift": 0.0011441752785116082}, "Light Conditions": {"gap_ratio": 1.000505827688943, "clean_drift": 0.0011725508964888869}, "Objects Layout": {"gap_ratio": 0.985077398379196, "clean_drift": 0.0011384938266064916}, "Robot Initial States": {"gap_ratio": 1.0307620579099335, "clean_drift": 0.0011739165832693062}, "Sensor Noise": {"gap_ratio": 0.9864103533514378, "clean_drift": 0.0010430947711617528}}}

### Input control verified
All 24 raw/clean initial pairs have exactly identical image tensors, proprio and seeds; only prompt and saved action_chunk differ. Both-expert gate strength6 threshold.5 now applied to test the video-mediated route in addition to the action-only gate.

### Pass 25: both-expert context gate strength6 threshold.5
{"pass": 25, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.07313694677359617, "gap_ratio": 0.9684127061154245, "edited_raw_clean_gap": 0.07312585460068259, "clean_drift": 0.001314749409997826, "language_drift": 0.0013701033933072202, "raw_change": 0.014026058361548134, "correction_alignment": 0.26025390625, "fraction_corrected": 0.048334430903196335, "by_category": {"Background Textures": {"gap_ratio": 0.9245938692441799, "clean_drift": 0.0013170697583274695}, "Camera Viewpoints": {"gap_ratio": 0.9901962508153938, "clean_drift": 0.0012669956186416068}, "Light Conditions": {"gap_ratio": 0.96212508797672, "clean_drift": 0.001299164367713198}, "Objects Layout": {"gap_ratio": 0.9879132736624359, "clean_drift": 0.0013032813056965667}, "Robot Initial States": {"gap_ratio": 0.9971943412885376, "clean_drift": 0.0013901358608304647}, "Sensor Noise": {"gap_ratio": 0.9431524562934992, "clean_drift": 0.0013086624824221662}}}

### Candidate: rank-24 oblique text contrast removal
Only action/video text_embedding.0.weight changed. P=I-BU^T, U^T B=I. B contains full paired contrast modes, U their clean/proprio-orthogonal detectors. This is an explicit idempotent geometric projector, no loss fitting or optimization. Both attention-gate candidates retained as reproducible metadata, not combined with this test.

### Pass 26: rank-24 oblique text removal
{"pass": 26, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.07312558771403775, "gap_ratio": 0.968262299814763, "edited_raw_clean_gap": 0.07276287990677385, "clean_drift": 0.0028939505810299795, "language_drift": 0.003840734477095724, "raw_change": 0.008010355899888017, "correction_alignment": 0.3475109934806824, "fraction_corrected": 0.036859042942523956, "by_category": {"Background Textures": {"gap_ratio": 0.953420805271654, "clean_drift": 0.0033089007024188957}, "Camera Viewpoints": {"gap_ratio": 0.9952067779683993, "clean_drift": 0.0028138868154615805}, "Light Conditions": {"gap_ratio": 0.8623834736473299, "clean_drift": 0.0022568853026676363}, "Objects Layout": {"gap_ratio": 0.9216491108957477, "clean_drift": 0.002437215853281822}, "Robot Initial States": {"gap_ratio": 0.9744113479947838, "clean_drift": 0.0028440363564959095}, "Sensor Noise": {"gap_ratio": 0.9545392769872695, "clean_drift": 0.0035015456218661215}}}

### Stronger selective gate candidate
Both experts, threshold lowered .5->.25, detector gain32->128, attention strength6 unchanged. This covers more low-amplitude tag-associated context responses; cleaner threshold reduces GELU negative tails. Rank-24 oblique candidate retained in metadata but removed from active edits.

### Pass 27: both-expert gate threshold.25, strength6, detector gain128
{"pass": 27, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.09678857857687263, "gap_ratio": 1.2815860305305145, "edited_raw_clean_gap": 0.0967912787922605, "clean_drift": 0.002367221268691695, "language_drift": 0.0032095255595020137, "raw_change": 0.054195503284436876, "correction_alignment": -0.08883852511644363, "fraction_corrected": -0.06375119090080261, "by_category": {"Background Textures": {"gap_ratio": 1.1495547956555552, "clean_drift": 0.0019268881803591487}, "Camera Viewpoints": {"gap_ratio": 1.270789650544649, "clean_drift": 0.0017100696864097013}, "Light Conditions": {"gap_ratio": 1.0684145906305602, "clean_drift": 0.0015539412406354097}, "Objects Layout": {"gap_ratio": 0.9940998316565113, "clean_drift": 0.0033824989088798935}, "Robot Initial States": {"gap_ratio": 1.4622969943039617, "clean_drift": 0.0030236427310785605}, "Sensor Noise": {"gap_ratio": 1.2516719636724636, "clean_drift": 0.001996681047280044}}}

### Selection for development rollout
Threshold.25 gate rejected: tagged action gap ratio1.2816 (worse). Conservative both-expert threshold.5/strength6/gain32 gate restored for a paired 96-episode development rollout. No candidate yet has success-rate validation.

### Pass 28: conservative gate development rollout
96 paired episodes: original51/96, edited52/96; +1.04pp, paired SE3.5pp; 6 wrong->right,5 right->wrong. Not convincing; do not retain on this evidence. Restored rank24 full oblique text edit (2 tensors) for identical development rollout.

### Pass 29: rank24 oblique edit development rollout
96 paired episodes: original50/96, edited50/96; 0pp, paired SE3.0pp; 4 wrong->right and4 right->wrong. No significant success gain; not validated. 11 passes remain. Continuing with existing token-detector readouts; preserve independent tagged, strip and language validation budgets.

### Temporary masked-context diagnostic before pass30
Reserved residual coordinates and affine anchors route action/video post-text-MLP context means into action self-attention summaries. The high-confidence suffix detector masks context tokens with u.x>.5. Read layers1-8 attn_out on the same80 inputs. ALL instrumentation must be removed immediately after acquisition. This is not a repair.

### Pass30 masked-context readout completed
Instrumentation fully removed. Readout time variation: action2.1e-5 RMS, video0. Masked post-text-MLP raw-clean mean contrast norms: action.128,video.311. Rank4 oblique output projector now applied to text_embedding.2.weight in both experts only. Protection includes56 clean/paraphrase means, linear proprio span, bias directions, and exact two-direction readout substitution span. No training; static SVD projector. Candidate awaits probe31.

### Pass31 masked-prefix rank4 output projection
{"pass": 31, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.08335778614090263, "gap_ratio": 1.103747733719252, "edited_raw_clean_gap": 0.07620919049243258, "clean_drift": 0.02955929519327312, "language_drift": 0.009596781246564786, "raw_change": 0.020384200647748252, "correction_alignment": -0.26936522126197815, "fraction_corrected": -0.07270410656929016, "by_category": {"Background Textures": {"gap_ratio": 1.1260176657630105, "clean_drift": 0.01065508244657377}, "Camera Viewpoints": {"gap_ratio": 1.049945945234569, "clean_drift": 0.010917807250397525}, "Light Conditions": {"gap_ratio": 1.2330178296478553, "clean_drift": 0.009471727550503175}, "Objects Layout": {"gap_ratio": 0.9531658972299529, "clean_drift": 0.010534382233567395}, "Robot Initial States": {"gap_ratio": 1.1361435740379058, "clean_drift": 0.06860767820286667}, "Sensor Noise": {"gap_ratio": 1.1322742367087602, "clean_drift": 0.010103561580763004}}}

### Prefix output projector rejected
Pass31 tag gap ratio1.1037, clean drift.02956, language drift.00960. Both edited tensors removed. Next candidate: four selective detector gates (common threshold.5, background.625, light.875, layout.625), gain128, attention strength12, both experts. Built from pass23 readout without fitting; no prefix projector combined.

### Pass32 four-detector gate strength12
{"pass": 32, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.07721051326298077, "gap_ratio": 1.0223511561267153, "edited_raw_clean_gap": 0.07715804881774666, "clean_drift": 0.0013573638458608942, "language_drift": 0.0014668580188027184, "raw_change": 0.02132139334144672, "correction_alignment": 0.061104413121938705, "fraction_corrected": 0.01725090481340885, "by_category": {"Background Textures": {"gap_ratio": 1.8667093526178637, "clean_drift": 0.0014817225652362133}, "Camera Viewpoints": {"gap_ratio": 0.9916666545607451, "clean_drift": 0.0012784354967895356}, "Light Conditions": {"gap_ratio": 0.9994218333400792, "clean_drift": 0.0012547421827180704}, "Objects Layout": {"gap_ratio": 1.0364714832328255, "clean_drift": 0.0013416075444253252}, "Robot Initial States": {"gap_ratio": 0.994049581120243, "clean_drift": 0.0014363151169599563}, "Sensor Noise": {"gap_ratio": 0.9413403143161695, "clean_drift": 0.0013369482850735935}}}

### Four-detector gate rejected after pass32
Tag action gap ratio1.0224; background ratio1.8667. Clean/language drift .00136/.00147, but no overall tag improvement. Restored the original single-detector architecture, changed only threshold.5->1.0 (strength6,gain32,both experts), for an isolated selectivity test.

### Pass33 single-detector threshold1.0, strength6
l05 accidentally duplicated in request; removed input index54 from analysis. {"pass": 33, "baseline_tag_gap": 0.07552249811649933, "raw_to_original_clean": 0.07293166659869448, "gap_ratio": 0.9656945733732444, "edited_raw_clean_gap": 0.0729226637408663, "clean_drift": 0.0013127165593731487, "language_drift": 0.0013690550659936338, "raw_change": 0.010532646078279839, "correction_alignment": 0.3114936649799347, "fraction_corrected": 0.04344205930829048, "by_category": {"Background Textures": {"gap_ratio": 1.0008580801468843, "clean_drift": 0.0013170697583274695}, "Camera Viewpoints": {"gap_ratio": 0.9838141152099097, "clean_drift": 0.0012669956186416068}, "Light Conditions": {"gap_ratio": 1.0020671939038055, "clean_drift": 0.001299164367713198}, "Objects Layout": {"gap_ratio": 1.001265552354887, "clean_drift": 0.0013345526339087047}, "Robot Initial States": {"gap_ratio": 0.963996081724653, "clean_drift": 0.0013901358608304647}, "Sensor Noise": {"gap_ratio": 0.945236115293811, "clean_drift": 0.0012641445514583781}}}

### Candidate frozen after pass33
Single detector threshold1.0,strength6,gain32,key64,both experts;308 tensors. Immutable copy in fwd/probe_33_weights. No further tuning. Planned remaining seven passes: raw validationA/B/C (384 paired), stripA/B (256 paired), languageA/B (256 paired). Retain only with independent positive success evidence and no material clean/paraphrase loss; otherwise restore original.

### Pass34 frozen-gate validation A (raw)
{"passes": [34], "n": 128, "original_success": 61, "edited_success": 63, "wrong_to_right": 6, "right_to_wrong": 4, "diff_pp": 1.5625, "se_pp": 2.476358441722394, "ci95_pp": [-3.291162545775892, 6.416162545775892]}
Not significant. Gate remains frozen. validationC selected by fixed hash frozen-validation-C-100000, category/suite balanced, disjoint from every previously used and reserved variant. Only practice_variants.json read.

### Pass35 frozen-gate validation B (raw)
{"passes": [35], "n": 128, "original_success": 53, "edited_success": 54, "wrong_to_right": 5, "right_to_wrong": 4, "diff_pp": 0.78125, "se_pp": 2.3519377978636435, "ci95_pp": [-3.8285480838127413, 5.391048083812741]}
Raw A+B: {"passes": [34, 35], "n": 256, "original_success": 114, "edited_success": 117, "wrong_to_right": 11, "right_to_wrong": 8, "diff_pp": 1.171875, "se_pp": 1.7044511611433868, "ci95_pp": [-2.1688492758410383, 4.512599275841039]}
### Passes36-37 frozen-gate clean checks (strip A+B)
{"passes": [36, 37], "n": 256, "original_success": 168, "edited_success": 166, "wrong_to_right": 8, "right_to_wrong": 10, "diff_pp": -0.78125, "se_pp": 1.6598070386756574, "ci95_pp": [-4.034471795804288, 2.4719717958042886]}
Candidate still frozen. Remaining: languageA/B, rawC. No significant tagged success improvement so far; final retention criteria unchanged.

### Pass38 frozen-gate language validation A
{"passes": [38], "n": 128, "original_success": 98, "edited_success": 95, "wrong_to_right": 1, "right_to_wrong": 4, "diff_pp": -2.34375, "se_pp": 1.741417279470994, "ci95_pp": [-5.756927867763148, 1.0694278677631481]}
Paraphrase point estimate -2.34pp; no gate tuning or weight changes. Complete languageB and rawC before final retain/revert decision.

### Pass39 frozen-gate language validation B; safety criterion failed
{"passes": [39], "n": 128, "original_success": 85, "edited_success": 80, "wrong_to_right": 4, "right_to_wrong": 9, "diff_pp": -3.90625, "se_pp": 2.806581425635193, "ci95_pp": [-9.407149594244977, 1.594649594244978]}
Pooled language A+B: {"passes": [38, 39], "n": 256, "original_success": 183, "edited_success": 175, "wrong_to_right": 5, "right_to_wrong": 13, "diff_pp": -3.125, "se_pp": 1.648956146338956, "ci95_pp": [-6.356954046824354, 0.10695404682435372]}
Language loss3.125pp exceeds approximately2pp allowed loss. Frozen gate will NOT be retained. Pass40 will complete preplanned rawC; then clear ALL edits and submit original weights. No other candidate had a validated significant tagged success gain.
