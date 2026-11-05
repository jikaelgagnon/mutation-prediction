# Retained upstream implementation

Source: Kuhlman-Lab/ThermoMPNN-D, revision
`df9a75aaddb674a7c4c193005031fc0536d325fb` (the baseline checkpoint reference).
The local copy has adaptations and is not an untouched upstream checkout.
The [MIT license](thermompnn/LICENSE) also covers code extracted into `data/`
and `model/metrics.py`; retain it when redistributing those adaptations.

Retained modules implement the ProteinMPNN encoder, structure parsing,
single-mutant ThermoMPNN v2 head, and their geometry/feature dependencies.
The nested upstream layout is preserved to make comparison with upstream easier.
Side-chain and IPMP helpers remain because retained modules import them;
the default workflow uses backbone features.

Local adaptations include package-relative namespace changes, tensor shape
annotations, injectable encoders/heads, and the single-mutant inference wrapper
in the parent package. Cleanup removes the Siamese class and upstream training,
benchmark inference, and ProteinMPNN training scripts. Parameter names in the
retained models are unchanged so state-dict checkpoints remain compatible.

Project-owned MegaScale loading and collation now live in `data/`; training
metrics live in `model/metrics.py`. Only measured train/val/test splits remain.
The old `mutation_prediction.thermo_mpnn_d` import namespace and
`thermompnn-train` command are retired. Use the public baseline wrapper and
project model API rather than importing vendor internals in application code.
