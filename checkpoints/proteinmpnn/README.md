# ProteinMPNN checkpoints

These files are pretrained **full-backbone ProteinMPNN** checkpoints from
`Kuhlman-Lab/ThermoMPNN-D/vanilla_model_weights`.

The upstream ProteinMPNN README lists these full-backbone checkpoint names
and explicitly explains `v_48_010` as using 48 edges and 0.10 Å coordinate
noise. Interpreting the other suffixes as hundredths of an angstrom follows
that naming pattern; the upstream README does not individually document each
checkpoint's noise value.

| File | Neighbors | Noise level |
| --- | ---: | ---: |
| `v_48_002.pt` | 48 | inferred: 0.02 Å |
| `v_48_010.pt` | 48 | 0.10 Å (documented upstream) |
| `v_48_020.pt` | 48 | inferred: 0.20 Å |
| `v_48_030.pt` | 48 | inferred: 0.30 Å |

The filename structure can be read as `v_<neighbors>_<noise>.pt`:

- `48` is the number of neighboring residues used by the model.
- The final three digits appear to encode the noise level in hundredths of an
  angstrom, based on the explicitly documented `010` example.

In ProteinMPNN, backbone noise is Gaussian perturbation applied to coordinates
during training. These files are alternative pretrained encoders, not
different ThermoMPNN-D prediction heads. ThermoMPNN-D's default is
`v_48_020.pt`; set `model.version` in the config to select another checkpoint.

Sources:

- [ProteinMPNN README](https://github.com/dauparas/ProteinMPNN), which lists
  the checkpoint names and documents `v_48_010` as 48 edges and 0.10 Å noise.
- [ThermoMPNN-D vanilla weights](https://github.com/Kuhlman-Lab/ThermoMPNN-D/tree/main/vanilla_model_weights).
