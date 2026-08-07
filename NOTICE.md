# Provenance and Attribution

TIAE (Topological Intent-Action Embedding) is an original intent-to-action
learning method developed by Molin Wu (molyswu), extending the JEPA-based
world-modeling line of work.

The research builds on the latent world-modeling and evaluation ecosystem of:

- **LeWM / LeWorldModel** - Lucas Maes, Quentin Le Lidec, Damien Scieur,
  Yann LeCun, and Randall Balestriero.
  <https://github.com/lucas-maes/le-wm>
- **stable-worldmodel** - the reusable world-modeling platform used by the
  LeWM ecosystem. <https://github.com/galilai-group/stable-worldmodel>
- **CLEAR-LeWM** - the separately versioned evaluator used only where results
  are explicitly labelled CLEAR. <https://github.com/cLEAR-LeWM/cLEAR-LeWM>

TIAE re-derives and re-implements the search-free intent-to-action operator
on top of a JEPA backbone, adds a topology-preserving distance loss between
the intent and action manifolds, and introduces a dual-channel state/intent
encoding scheme with cross-attention fusion. The previous codebase from
the upstream TIAE-JEPA project by Junhan Sun, Hao Zhao, and Guofeng Zhang
(at zju3dv/TIAE-JEPA) served as the architectural starting point and is
preserved here as a baseline reference.

The original comparison used LeWM revision
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. LeWM-derived files include
`module.py`, `jepa.py`, `train.py`, `eval.py`, `utils.py`, and their Hydra
configurations. TIAE adds a topology-preserving loss, a shared
intent-to-action operator, dual-channel encoding, paired physical/goal
action-law supervision, shared-encoder multi-task training, and Direct
control with O(1) inference. The applicable upstream MIT notice is
preserved in `third_party/LeWM-MIT-LICENSE.txt`.

This repository is distributed under the MIT License and contains
project-authored documentation, visual assets, and released training and
evaluation source. Third-party datasets, CLEAR-LeWM, simulator assets, and
model checkpoints are not relicensed by this repository. Consult each
artifact's own license and terms before use or redistribution.

External methods mentioned in result tables retain their own licenses,
protocols, and citations. Their reported values are comparison context and do
not imply redistribution of their source or checkpoints.

The project website vendors Three.js 0.185.1 and OrbitControls under the MIT
License. Its unmodified license text is preserved at
`docs/vendor/three/LICENSE.txt`; the OrbitControls module import is adjusted
only to resolve the vendored local module.
