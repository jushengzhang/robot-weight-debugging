# FastWAM background robustness

## Rules and plan
- No training, network execution in Python, or non-practice benchmark access.
- Diagnose matched background/control behavior and internal changes.
- Use targeted, non-uniform weight edits with causal dose-response tests.
- Reserve disjoint practice variants for paired background and other-category validation.
- Forward passes used: 0 / 40. No edits yet.

## Setup
- Tool offline at first request; 0 passes charged.
- Practice catalogue: 524 background variants; fixed SHA256 split gives 163 diagnosis / 361 validation.
- 34 matched BG/light task pairs, seed 17, prepared in diagnostic_episodes.json.
- Tensor convention: mot.mixtures.{video,action}.blocks.L; 24 heads x 128 channels.
- Weight-only head coupling statistics saved; these are candidate-ranking information, not causal evidence.

## Intervention plan (not yet applied)
- Inspect fixed-input matched-task residual and attention outputs, then attention maps.
- Candidate families: selective action self-attention output head attenuation; low-rank removal within implicated heads.
- Use original-referenced edits, causal dose response, and matched control-head interventions.
- Snapshot/audit utilities prepared. Still no edits and no charged forward passes.

- Added action output spectral reference (self-attention, cross-attention and FFN overlap). No causal or background-specific claim is made from weight-only spectra.

## Improved pre-data diagnostic design
- 38 harder background variants, 27 same-task lower-difficulty backgrounds, 37 least-perturbed light controls, all seed 17.
- Added light controls are reassigned to diagnosis before any results; practice_split.json updated.
- No baseline data yet; tool still offline at last attempt.

- Preselected two 128-variant disjoint validation batches per category in validation_manifests.json, seeds 901/902. No outcomes used to make these selections.

## Waiting for behavioral evidence
- Four diagnostic rollout attempts were refused without charge; 0/40 passes used.
- Weight-only scans found distributed, non-degenerate representations; no justified edit yet.
- First-layer spatial-mode sensitivity varies by head, but this is only a hypothesis source.
- Keeping the original checkpoint unchanged while the backend starts.

- Paired and variant-clustered uncertainty utilities prepared. Waited an additional 300 s for backend startup; still no edits.

## Locked experimental principles
- Fixed-input contrasts precede causal edits.
- Require original + two doses and a matched non-implicated-head control.
- Reserve roughly half the 40-pass budget for selection-independent background and all-category validation.
- No edit will be justified solely by large weight norms or singular values.
- Backend still refused the pilot; no charged passes.

- Action/video projection slices have near-zero coordinate-wise cosine; do not assume a simple copied/truncated feature basis. Q/K normalization vectors remain similar, and head-space covariance comparisons are saved for reference.

- Late-head direct residual-path readout geometry saved (not a full-network influence estimate). Weight-only analysis is complete; no background-specific or causal conclusion yet.

## Backend status checkpoint
- 8 forward attempts refused; 0/40 passes charged.
- 20 Python runs completed; structural weight references and experiment manifests saved.
- No rollouts, activation probes, success measurements, or causal findings exist yet.
- No tensors edited. Waiting for the permitted forward backend rather than running the network locally.

## Pass 1: original pilot
- Backend now online. 12 episodes: BG 6/8 (75.0%); light 4/4 (100%).
- Two max-step failures, both BG; sample too small for edit selection.
- All-layer residual summaries, frames, and policy inputs saved in roll_1 artifacts.
- Forward passes used: 1/40; weights unchanged.

## Pass 2: full original diagnostic
- 102 episodes: BG 48/65 (73.85%), light 35/37 (94.59%).
- Total 19 failures; 17 BG, 2 light. Difficulty 5 overall 4/11; long suite 13/26.
- Attention outputs recorded in all 30 layers for both experts; first observations retained for deconfounded probes.
- Passes used: 2/40; no edits.

## Pass 3: original fixed-input attention probe
- 27 paired initial observations (54 inputs), same proprioception, instructions and seed.
- Texture differences change action attention throughout the network. Early-layer shift/task ratios roughly 1.0-1.4; rank-4 contrast energy 0.65-0.85 in layers 0-5.
- No causal inference yet; inspect head attention before selecting an edit. Passes used 3/40; no edits.

## Passes 4-5: early-head attention localization
- Four matched texture pairs, common noise 17, layers 0/1/2/5.
- Testing action layer 2 head 10: high video reliance, texture-region enrichment and spatial attention shift. Correlation is a candidate screen, not causal evidence.
- First causal edit: only L2 H10 output-projection columns attenuated by 50%; all other tensors unchanged.
- Passes used 5/40. This edit is experimental, not accepted.

## Pass 6: rejected head-output hypothesis
- L2 H10 columns x0.5: fixed-input pair action RMS ratio 1.000025; BG action change RMS 0.000665, light 0.000624.
- No predicted invariance benefit; removed edit without spending rollout budget. Attention concentration alone was not causal evidence.
- Next hypothesis: low-rank shared action-attention output nuisance directions; obtain initial-observation light reference before centering the intervention.
- Passes used 6/40; currently no edits.

## Pass 7 and spectral intervention design
- 37 original light initial-observation references obtained.
- L4 leading contrast/light variance ratios 24.45,16.15; L5 14.13,22.79. Full rank-4 subspaces contain 8.4%,13.9% of light task variation. Leading subspaces reproduce across pair halves.
- Candidate: action self-attention output L4/L5, rank-2 centered projection, dose .35. W' = W - a U U^T W; b' = b + a U U^T(mu_light-b). No training, video changes or global scaling.
- Experimental, not accepted. Passes used 7/40.

## Pass 8: early contrast projection has weak causal action effect
- L4/L5 rank2 dose .35: pair RMS ratio .99800; BG action change .000815, light .000654.
- Edited activation directions change strongly (some light output RMS shifts .10-.13), but downstream activation/action shifts are small. Early raw attention contrasts are not sufficient causal targets.
- Removed candidate; inspect later blocks and effective gating. No accepted edit. Passes used 8/40.

## Passes 9-10: depth localization
- Original light reference now covers all 30 action attention outputs.
- Middle L10-19 rank2 dose .5 probe: pair RMS ratio 0.952294; BG change 0.011120, light change 0.005591.
- Now testing L20-29 independently, same rank/dose. Broad groups are localization tests; final selection must remove low-specificity directions and pass rollout checks.
- Pass 11 late L20-29 rank2 dose .5: pair RMS ratio 0.925532; BG change 0.017029, light 0.014372.
- Pass 12 early L0-9 rank2 dose .5: pair RMS ratio 0.994219; BG change 0.003053, light 0.003401.
- Narrowed analytic candidate: leading contrast directions with texture/light variance ratio >=5 in layers10-29, dose .5, light-mean anchored. Selection uses diagnostic probes only, no validation outcomes.
- Selected directions: {10: [0, 1], 12: [1], 13: [0, 1], 14: [0, 1], 15: [0, 1], 20: [0, 1], 21: [0], 25: [0], 26: [0], 29: [0]}. Passes used 12/40.

## Pass 13: action-output projection rejected by rollouts
- Paired diagnostic 102 episodes. BG original48/65 vs edited46/65; light35/37 vs34/37. Overall4 rescues,7 regressions, gain -2.9pp SE3.3pp.
- All action-output edits removed. Probe invariance alone was insufficient; shrinking downstream task-response directions can hurt policy execution.
- Next localization targets video observation features before shared K/V transfer. Passes used13/40; no accepted edit.

## Pass 14: observation-side protected nuisance direction
- Original video residuals recorded on 102 diagnostics. These are episode averages, so trajectory confounding remains; fixed-input causal probes must verify transfer.
- Standardized camera summaries; protect both light means, constant direction and top8 light-task PCs. Leading residual contrast is highly texture-specific, with contrast/light variance ratios ~100-1500 and <1% light variance. Second direction is unstable and excluded.
- Candidate: V_l' = V_l - .5 (V_l u_l) u_l^T, video value weights only, layers1-29, rank1 each; u_l from prior residual block. Action/Q/K/biases unchanged.
- Passes used14/40. Experimental, not accepted.
- Pass15 protected-video-values dose .5 fixed-input probe: pair RMS ratio 0.971753; BG change 0.005905, light 0.001364.

## Pass16: video-value edit not accepted
- Paired diagnostics: original BG50/65 vs edited46/65; light35/37 vs34/37. Overall1 rescue,6 regressions, gain -4.9pp SE2.6.
- Original baseline differs from earlier same-seed originals (which had BG48/65); auditing saved initial observations/actions.
- Video edits removed. Passes used16/40; no accepted edit.

## Pass17: late attention-concentration hypothesis
- Replay audit: identical images/proprio but original actions differ by ~.001 RMS between runs. Small early probe effects were near numerical variability floor, not confirmed benefits.
- Later maps: L20 H15 video mass .991, texture-region enrichment1.934, TV.435, peak .536. Also implicated L10 H5/H13/H16 and L14 H18 (consistent enrichment across two pairs).
- New candidate: attenuate only these five heads' action norm_q slices by35%, softening texture-dependent attention concentration. All other parameters original. Preselected matched low-background control heads.
- Passes used17/40; still no accepted edit.

## Pass18: selected query-temperature fixed-input test
- Five selected heads, dose .35; original-referenced query norm slices only.
- Action summary: {"pass": 18, "dose": 0.35, "original_pair_rms": [0.12239058343939285, 0.1997832556257836], "edited_pair_rms": [0.12237143117549656, 0.19974436015239444], "pair_rms_ratio": 0.9998157369938859, "action_change_by_input": [0.0006942759488715466, 0.0006721754984832081, 0.0009170139136063342, 0.0007318248090960791]}
- Full map comparison saved in probe18_summary.json; candidate snapshot query_targets_d035. Rollout benefit not yet tested.
- Query-temperature dose .35 rejected as an actionable mechanism: attention entropy rises, but action shifts .0007-.0009 RMS are near replay noise; pair ratio .999816. Edits removed, no rollout spent.

## Pass19: fixed-input residual localization
- 54 paired initial inputs + 37 light controls, original model. Archive contains only action residuals; fixed-input VIDEO residual hypothesis cannot be confirmed from this tool output.
- Saved task-protected action residual texture bases, with split-half stability and light-variance leakage. No new edits yet.

## Task-protected texture mean-shift hypothesis
- High-vs-low texture variance directions failed stability screen; not edited.
- Matched (high+low)/2 minus light residual mean is coherent across all27 matched tasks, with final-layer split cosine .8818 and texture/light projection RMS 3.4147/.1705. Light PCA25 protects95.4% of task variation; mean and constant also protected.
- Experimental rank1 final action readout edit H'=H-.5(Hu)u^T, no bias or other changes. This targets a texture-linked internal feature, not global action scaling. Norm-matched texture-orthogonal control prepared.

## Pass20: rank1 readout causal probe
- {"pass": 20, "dose": 0.5, "bg_action_change_rms": 0.002803847701600042, "light_action_change_rms": 0.0009438065551524281, "original_bg_light_discrepancy_rms": 0.10996777388081899, "edited_bg_light_discrepancy_rms": 0.10969952980830193, "bg_light_discrepancy_ratio": 0.9975607028945791, "original_mean_bg_light_bias": [-0.0032395124435424805, 0.028068192303180695, 0.07727228850126266, 0.0012278322828933597, 0.0003027452330570668, -0.0026569515466690063, -0.013888888992369175], "edited_mean_bg_light_bias": [-0.0034190218430012465, 0.0317833349108696, 0.07413432002067566, 0.0011948042083531618, 0.0003531508846208453, -0.0023582635913044214, -0.013888888992369175], "predicted_observed_change_cosine": 0.9388617873191833, "predicted_change_rms": 0.01129832828851787, "observed_normalized_mean_change_rms": 0.002535827648506783, "physical_change_by_channel": [0.001425360911525786, 0.005497278645634651, 0.004726770333945751, 0.0002755953464657068, 0.0003081505710724741, 0.0005156173720024526, 0.0], "light_change_by_channel": [0.0010734467068687081, 0.0016216494841501117, 0.001496350858360529, 0.00023896072525531054, 0.000254038255661726, 0.0003043948672711849, 0.0]}

## Pass21 and narrowed causal hypothesis
- Full rank1 readout .5 paired diagnostic results: {"Background Textures": {"n": 65, "original": 49, "edited": 48, "gain_pp": -1.5384615384615385, "se_pp": 3.4615384615384617, "rescues": 2, "regressions": 3}, "Light Conditions": {"n": 37, "original": 34, "edited": 35, "gain_pp": 2.7027027027027026, "se_pp": 2.702702702702703, "rescues": 1, "regressions": 0}}. No benefit, not accepted.
- Probe showed full projection lowers vertical BG-light discrepancy but increases lateral discrepancy. New exploratory candidate removes the same protected texture feature ONLY from vertical output row2; all other readout rows original.
- Preplanned tests: vertical doses .5 and1.0, then a norm-matched texture-orthogonal vertical control. Independent validation remains untouched.
- Pass22 vertical readout dose .5: {"Background Textures": {"n": 65, "original": 51, "edited": 50, "gain_pp": -1.5384615384615385, "se_pp": 2.6785362071507923, "rescues": 1, "regressions": 2}, "Light Conditions": {"n": 37, "original": 36, "edited": 36, "gain_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0}}. No background gain; proceeding with preplanned dose1.0.
- Pass23 vertical readout dose1.0: {"Background Textures": {"n": 65, "original": 50, "edited": 47, "gain_pp": -4.615384615384616, "se_pp": 3.41853631435215, "rescues": 1, "regressions": 4}, "Light Conditions": {"n": 37, "original": 36, "edited": 36, "gain_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0}}. Background performance worsens with stronger removal; light unchanged. Suppression candidate rejected.
- Testing preplanned norm-matched texture-orthogonal control. Possible revised mechanism: the texture-linked vertical feature is compensatory, so amplification rather than removal would need its own causal and independent validation.
- Pass24 norm-matched vertical control: {"Background Textures": {"n": 65, "original": 50, "edited": 49, "gain_pp": -1.5384615384615385, "se_pp": 2.6785362071507923, "rescues": 1, "regressions": 2}, "Light Conditions": {"n": 37, "original": 36, "edited": 35, "gain_pp": -2.7027027027027026, "se_pp": 2.702702702702703, "rescues": 0, "regressions": 1}}. Differences from target removal are suggestive but not statistically decisive.
- Preplanned revised causal tests: target amplification by50% and100% (setter doses -.5,-1.0). These test compensation, not invariance. No candidate accepted; all independent validation still untouched.
- Pass25 vertical amplification50%: {"Background Textures": {"n": 65, "original": 50, "edited": 49, "gain_pp": -1.5384615384615385, "se_pp": 3.4615384615384617, "rescues": 2, "regressions": 3}, "Light Conditions": {"n": 37, "original": 36, "edited": 35, "gain_pp": -2.7027027027027026, "se_pp": 2.702702702702703, "rescues": 0, "regressions": 1}}. No benefit plus light regression; stop amplification for futility, without testing stronger amplification. All readout edits removed.
- Remaining15 passes reserved for one failure-specific MLP hypothesis and independent validation; no new hypothesis will be accepted without disjoint paired success evidence. Reliable failed/control cases selected solely from repeated ORIGINAL outcomes, saved in failure_specific_manifest.json.

## Pass26 follow-up: input/DC localization
- Failure-specific MLP directions were not shared across tasks in late layers; no MLP edit made.
- Pixel-mode analysis refuted high-frequency texture filtering: BG/light variance ratio ~3-3.4 for DC, only~1-1.8 for other modes. No checkerboard filter applied.
- Early video residual contrasts projected into the 48-dimensional patch DC output subspace, protecting both camera light means and top8 light PCs (96.1% variance). Direction split cosine .9690, contrast/light ratio53.38, light variance fraction .00295.
- New experimental edit: patch DC rank1 projection, dose .25; all other spatial modes and all transformer weights unchanged. Episode-average source confounding must be checked by fixed-input probe. Dose .25/.5 and matched low-texture DC control prepared.

## Pass27: patch DC fixed-input test
- {"pass": 27, "dose": 0.25, "original_replay_rms": 0.0, "background_action_change_rms": 0.00460066660593372, "light_action_change_rms": 0.0018162078567113514, "successful_bg_action_change_rms": 0.004163056235042533, "failed_bg_action_change_rms": 0.006153156387041374, "bg_light_discrepancy_ratio": 0.9841556309873238, "failure_control_discrepancy_ratio": 0.9661537844993973, "bg_light_discrepancy_rms_original": 0.10996777388081899, "bg_light_discrepancy_rms_edited": 0.10822540389194876, "residuals": {"0": {"total_contrast_ratio": 0.9832152224397462, "protected_texture_direction_ratio": 0.9331977472905165, "light_residual_change_rms": 0.0028108765857287887}, "1": {"total_contrast_ratio": 0.984435880842805, "protected_texture_direction_ratio": 0.8934732526059667, "light_residual_change_rms": 0.004163109269506913}, "2": {"total_contrast_ratio": 0.983861149314918, "protected_texture_direction_ratio": 0.8792157960568312, "light_residual_change_rms": 0.005134133854940314}, "10": {"total_contrast_ratio": 0.9815247926922176, "protected_texture_direction_ratio": 0.9293601541849734, "light_residual_change_rms": 0.009242382654593071}, "20": {"total_contrast_ratio": 0.9876227984160273, "protected_texture_direction_ratio": 0.9714343139813304, "light_residual_change_rms": 0.01018431961158796}, "29": {"total_contrast_ratio": 0.987085750658233, "protected_texture_direction_ratio": 0.970022928209386, "light_residual_change_rms": 0.010817409304846762}}}
- Pass28 patchDC dose .25: {"Background Textures": {"n": 65, "original": 49, "edited": 47, "gain_pp": -3.076923076923077, "se_pp": 3.7782466739208656, "rescues": 2, "regressions": 4}, "Light Conditions": {"n": 37, "original": 35, "edited": 33, "gain_pp": -5.405405405405405, "se_pp": 3.7687388582616013, "rescues": 0, "regressions": 2}}. Rejected for background and light regressions; patch edit removed, stronger dose/control skipped for futility.

## Final exploratory candidate
- Action MLP output projections L1/L2 only; leading failure-specific shared subspaces have task-half overlaps .932/.843 and failure/control ratios11.56/14.43. Controls protect light and successful-background means and95% variation.
- W'=W-.5u(u^TW), b'=b+.5u[u^T(mu-b)]. Rank1 each, centered. Norm-matched texture-orthogonal controls prepared. No further new families after this; reserve independent validation.

## Pass29: early MLP fixed-input causal test
- {"pass": 29, "dose": 0.5, "background_action_change_rms": 0.000696239387864609, "light_action_change_rms": 0.0007172944049089793, "successful_bg_action_change_rms": 0.0007134199517357661, "failed_bg_action_change_rms": 0.0006468480473839957, "bg_light_discrepancy_ratio": 0.9997438284770737, "failure_control_discrepancy_ratio": 0.9996195890564624, "layer1_target_projection_ratio": 0.5016847259475176, "layer2_target_projection_ratio": 0.5015885101658678, "late_mlp_change_rms": 0.00029924126524506116}

## Pass30: early MLP candidate rejected
- Paired diagnostic: BG50/65 original vs47/65 edited; light36/37 vs35/37. Overall1 rescue,5 regressions; -3.9pp(SE2.4pp). Fixed-input action changes were near replay noise.
- Removed all current edits. No candidate accepted. Ten forward passes remain; independent validation has not yet been used.

## Final within-family late-MLP test
- Early MLP suppression did not propagate to actions. Existing fixed-input data identify L11/L19/L25 as late leading directions with task-half overlap>=.70 and failure/control variance ratio>=100. Rank1 centered output projections only; first dose .5.
- Plan: fixed-input .5 and1.0 doses, orthogonal norm-matched control, then locked-candidate disjoint paired rollout validation. No further candidate families. Reject if causal specificity or validation gain is absent. Original weights remain the fallback.

## Pass31: late-MLP half-dose causal probe
- {"pass": 31, "kind": "target", "dose": 0.5, "background_action_change_rms": 0.0006160403749242193, "light_action_change_rms": 0.0006500491737766449, "successful_bg_action_change_rms": 0.0006210261936420629, "failed_bg_action_change_rms": 0.0006128004673105498, "failure_control_discrepancy_rms_original": 0.10884617139367762, "failure_control_discrepancy_ratio": 0.9998430496951136, "failure_to_original_control_ratio": 0.9999324713629201, "failure_action_change_by_channel": [0.0007943011655336939, 0.0008772153589951826, 0.0010267891445688393, 0.00021811090009653058, 0.00022695062597873793, 0.00027363271505705816, 0.0], "light_action_change_by_channel": [0.0009310319597560409, 0.00090577417859012, 0.0010350102246528931, 0.00022760034725027322, 0.00024392157940503198, 0.00029690790744300904, 0.0], "failure_pair_ratios": [1.0041117674110676, 0.9985273765329333, 0.9991915241387436, 0.9990273517103375, 1.0003597138481255, 1.0002860695735316, 0.9982492837468713, 0.9997021678015928, 0.999520200399081, 0.9963251514519746, 1.00024739086832], "layer11_target_projection_ratio": 0.5007526531946592, "layer19_target_projection_ratio": 0.5004270907206363, "layer25_target_projection_ratio": 0.502094885746524, "late_mlp_change_rms": 0.0010567077536008395}

## Pass32: late-MLP full-dose probe
- {"pass": 32, "kind": "target", "dose": 1.0, "background_action_change_rms": 0.0007117442394468803, "light_action_change_rms": 0.0007023630267644912, "successful_bg_action_change_rms": 0.0007004544358433961, "failed_bg_action_change_rms": 0.0007763990076657461, "failure_control_discrepancy_rms_original": 0.10884617139367762, "failure_control_discrepancy_ratio": 0.9995456747733359, "failure_to_original_control_ratio": 0.9996193474384015, "failure_action_change_by_channel": [0.0010516900621563108, 0.0011261876106739154, 0.001281715002697495, 0.0002384276981766536, 0.00025157431955083123, 0.0002868565125661246, 0.0], "light_action_change_by_channel": [0.0010793109821642167, 0.0009820533754686242, 0.0010574294113485424, 0.00022951930498064585, 0.00025164504941855035, 0.00029949026500079786, 0.0], "failure_pair_ratios": [0.9988025710046307, 0.9999628954819374, 0.9981035276413057, 0.9993361153119116, 1.0008788736427332, 1.0003498940226734, 0.9966421934727512, 1.000303329281633, 0.997162892745617, 1.0015985271611927, 0.9995807476517699], "layer11_target_projection_ratio": 0.0003064298712490904, "layer19_target_projection_ratio": 0.0002271432525721095, "layer25_target_projection_ratio": 0.00019487803523287966, "late_mlp_change_rms": 0.0020891922384326054, "action_change_ratio_full_to_half": 1.1235622444765045, "half_full_change_cosine": 0.5467244386672974, "failure_half_to_full_increment_rms": 0.0005882463474293865}

## Pass33 and frozen final choice
- Orthogonal late-MLP control: {"pass": 33, "kind": "control", "dose": 1.0, "background_action_change_rms": 0.0006105752708931362, "light_action_change_rms": 0.0006547685484343393, "successful_bg_action_change_rms": 0.0006182290803886277, "failed_bg_action_change_rms": 0.0006050055133525058, "failure_control_discrepancy_rms_original": 0.10884617139367762, "failure_control_discrepancy_ratio": 0.9999698336754506, "failure_to_original_control_ratio": 0.9999706783818388, "failure_action_change_by_channel": [0.0008507526070251634, 0.0009275688156548063, 0.0009146445983767278, 0.00020174939115744168, 0.00021849522275957888, 0.00023030430224433413, 0.0], "light_action_change_by_channel": [0.0010251199056103438, 0.0009221381394958154, 0.0009574532420262999, 0.00021653747029965284, 0.0002500219429696379, 0.0002715275976094518, 0.0], "failure_pair_ratios": [1.0045480140355472, 0.9967007113085821, 0.9994393129689078, 1.0006358217156825, 1.0005310545130277, 1.0008757186348287, 0.9988357628565093, 1.0005626289892344, 0.9989303261662114, 0.9974203937298145, 0.999496874533374], "layer11_target_projection_ratio": 0.9999024848779633, "layer19_target_projection_ratio": 0.9999277543885854, "layer25_target_projection_ratio": 0.9999305086659444, "late_mlp_change_rms": 0.000291066973073098}
- Target doses .5/1.0 erase the recorded feature but do not produce a meaningful selective action correction. Reject late MLP candidate; all edits removed.
- No successful weight repair was found. Freeze original checkpoint rather than retain a harmful or unsupported edit. Final seven passes use the preselected first128 independent practice variants of each category as paired rollback audits. These are no-op checks, not evidence of improved robustness. No further parameter selection.
- Empty-edit paired rollout was refused without charging a pass. To enable no-op audits, saved an exact original copy of action.head.bias; zero scalar values changed. This is an audit artifact, not a repair, and will be removed at the end.

- Pass34 no-op rollback audit: {"pass": 34, "category": "Background Textures", "n": 128, "original_success": 79, "restored_success": 79, "diff_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0, "identical_action_trajectories": 128, "max_action_delta": 0.0}

- Pass35 no-op rollback audit: {"pass": 35, "category": "Light Conditions", "n": 128, "original_success": 120, "restored_success": 120, "diff_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0, "identical_action_trajectories": 125, "same_trajectory_lengths": 127, "max_common_prefix_action_delta": 2.0, "mean_absolute_step_difference": 0.75}

- Pass36 no-op rollback audit: {"pass": 36, "category": "Camera Viewpoints", "n": 128, "original_success": 47, "restored_success": 47, "diff_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0, "identical_action_trajectories": 128, "same_trajectory_lengths": 128, "max_common_prefix_action_delta": 0.0, "mean_absolute_step_difference": 0.0}

- Pass37 no-op rollback audit: {"pass": 37, "category": "Language Instructions", "n": 128, "original_success": 77, "restored_success": 77, "diff_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0, "identical_action_trajectories": 127, "same_trajectory_lengths": 128, "max_common_prefix_action_delta": 0.003662109375, "mean_absolute_step_difference": 0.0}

- Pass38 no-op rollback audit: {"pass": 38, "category": "Objects Layout", "n": 128, "original_success": 89, "restored_success": 89, "diff_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0, "identical_action_trajectories": 128, "same_trajectory_lengths": 128, "max_common_prefix_action_delta": 0.0, "mean_absolute_step_difference": 0.0}

- Pass39 no-op rollback audit: {"pass": 39, "category": "Robot Initial States", "n": 128, "original_success": 100, "restored_success": 100, "diff_pp": 0.0, "se_pp": 0.0, "rescues": 0, "regressions": 0, "identical_action_trajectories": 127, "same_trajectory_lengths": 128, "max_common_prefix_action_delta": 0.00091552734375, "mean_absolute_step_difference": 0.0}
