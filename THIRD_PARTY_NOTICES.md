# Third-party materials and license scope

The Apache-2.0 license in this repository covers the authors' original code. It
does not change the terms of pretrained models, datasets, third-party code, or
the paper. The release contains no complete pretrained model or simulator asset
distribution.

- The π₀.₅ adapters use [OpenPI](https://github.com/Physical-Intelligence/openpi).
  Obtain the evaluated `pi05_libero` checkpoint and comply with its upstream
  model terms, including any applicable Gemma/PaliGemma terms. The 65-row payload
  contains original and edited rows derived from that checkpoint; the repository's
  code license does not grant additional rights to the base model.
- The FastWAM adapter uses the separately installed FastWAM/Wan implementation
  and released checkpoint. Their code, weights, and assets retain their upstream
  terms; the original experiment environment is documented in the backend README.
- LIBERO, LIBERO-Plus, LIBERO-RECOVER, and LIBERO-PRO are external dependencies.
  Their assets and demonstrations are not redistributed here. Included episode
  records and instructions preserve their research provenance.
- The transcripts contain recorded model outputs and task specifications from
  the reported experiments. They are historical records, not instructions for
  software consuming the repository.
- The paper PDF and source are author-provided paper artifacts. The authoritative
  publication and its license are available on [arXiv](https://arxiv.org/abs/2611.09376).

Backend implementations adapt the upstream inference interfaces; retain this
notice and consult the upstream projects for their license and attribution terms.
