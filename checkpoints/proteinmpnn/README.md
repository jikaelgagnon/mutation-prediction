# ProteinMPNN checkpoints

These files are pretrained **full-backbone ProteinMPNN** checkpoints from
`Kuhlman-Lab/ThermoMPNN-D/vanilla_model_weights`.

The filenames follow the pattern `v_<neighbors>_<noise>.pt`:

- `48` is the number of neighboring residues used by the model.
- The final three digits give the backbone-coordinate noise level used during
  training, in hundredths of an angstrom.

| File | Neighbors | Training noise |
| --- | ---: | ---: |
| `v_48_002.pt` | 48 | 0.02 Å |
| `v_48_010.pt` | 48 | 0.10 Å |
| `v_48_020.pt` | 48 | 0.20 Å |
| `v_48_030.pt` | 48 | 0.30 Å |

The training noise is Gaussian perturbation applied to backbone coordinates
while training ProteinMPNN, to encourage robustness to structural variation.
These are alternative pretrained encoders, not different ThermoMPNN-D
prediction heads. ThermoMPNN-D's default is `v_48_020.pt`; set
`model.version` in the config to select another checkpoint.

These weights seem to come from the `vanilla_model_weights` folder from the [ProteinMPNN repo](https://github.com/dauparas/ProteinMPNN).
