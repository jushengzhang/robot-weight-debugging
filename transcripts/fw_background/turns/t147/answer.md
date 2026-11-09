**Final choice: preserve the original checkpoint.** No weight-value changes are retained. The only saved tensor is an exact original copy of `mot.mixtures.action.head.bias`, used to enable paired rollback audits.

The tested interventions did not justify submission:
- Texture-related attention and residual directions were measurable, but suppressing them did not reliably improve task success.
- The early-MLP candidate reduced diagnostic background success from **50/65 to 47/65** and lost one light-condition success.
- Full removal of the late-MLP directions nearly eliminated their measured activations, but reduced the failed/control action discrepancy by only **0.045%**, without a convincing selective behavioral effect.

The final rollback audits covered **896 reserved practice pairs**, 128 per category. Success outcomes matched in every pair: **0.0 percentage-point difference in all seven categories**. These were unchanged-weight checks, not evidence of improvement.

**I did not achieve the requested background-robustness gain within the budget.** Preserving the original weights avoids retaining an unsupported or harmful edit; I cannot honestly claim a successful repair.