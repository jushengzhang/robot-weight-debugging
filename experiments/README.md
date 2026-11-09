# Experiment code — first research release

These are the sources used for the paper's robot editing experiments. For
version 2 and real-robot work in progress, see [project status](../docs/PROJECT_STATUS.md).

| Component | Start here |
| --- | --- |
| LLM loop and persistent CPU Python kernel | [driver.py](harness/driver.py) · [kernel.py](harness/kernel.py) |
| π₀.₅ language task and forward tool | [Task rules](domains/pi05/domain_pi05.py) · [Forward tool](domains/pi05/pi05_forward.py) |
| Recovery task | [Task rules](domains/pi05/domain_recover.py) · [Backend](backends/pi05_recovery) |
| LIBERO-PRO task | [Task rules](domains/pi05/domain_pro.py) · [Backend](backends/pi05_libero_pro) |
| FastWAM perturbation and suffix tasks | [Perturbation rules](domains/fastwam/domain_fastwam.py) · [Suffix rules](domains/fastwam/domain_fastwam_suffix.py) |
| Policy servers and environment setup | [π₀.₅ backend](backends/pi05) · [FastWAM backend](backends/fastwam) |

The sources preserve the historical experiment layout. For a new machine, use
[`scripts/configure_workspace.py`](../scripts/configure_workspace.py) and follow
the [reproduction guide](../docs/REPRODUCING.md). The configuration tool relocates
paths; it does not install external simulators or download base-model weights.
The original machine layout is documented in [EXPERIMENT-SNAPSHOT.md](EXPERIMENT-SNAPSHOT.md).

The [evaluated sparse edit](../edits/pi05-language-c), [16 session transcripts](../transcripts),
and [raw evaluation evidence](../evidence) are included in this repository.
Download bundles are also available in the
[first release](https://github.com/jushengzhang/robot-weight-debugging/releases/tag/v0.1.0).
