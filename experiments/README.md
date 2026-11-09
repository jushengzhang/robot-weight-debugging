# Research code snapshot — November 5, 2026

This is the experiment code supplied for the paper, not a newly ported installation. `harness/` contains the LLM loop and CPU kernel; `domains/` contains each task specification and held-out evaluator; `backends/` contains the policy servers. Backend weights, environments, assets, and external simulator dependencies are not bundled. See EXPERIMENT-SNAPSHOT.md for the original host layout.

For a new machine, use `scripts/configure_workspace.py` from the repository root. These source files preserve the historical experiment layout; the public repository provides the code license and third-party notices. The sparse embedding edit is a separate download with an application script. All sixteen run transcripts and raw evaluation records are separate downloads.

Public code: https://github.com/jushengzhang/robot-weight-debugging
Paper: https://arxiv.org/abs/2611.09376
