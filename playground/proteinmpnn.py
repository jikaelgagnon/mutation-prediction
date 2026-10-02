# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.3.0",
#     "bio==1.8.4",
#     "marimo>=0.25.1",
#     "numpy==2.5.3",
#     "omegaconf==2.3.1",
#     "pandas==3.0.6",
#     "pytorch-lightning==2.6.6",
#     "torch==2.14.1",
# ]
# ///

"""Interactive trace of MegaScale double mutants through ProteinMPNN."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trace a MegaScale double mutant through ProteinMPNN

    Choose a MegaScale validation example and a pretrained ProteinMPNN checkpoint.
    This notebook follows the data through structure featurization and ProteinMPNN,
    then extracts the two mutation-site feature vectors used by
    `TransferModelv2Siamese`.

    ProteinMPNN is loaded as a **frozen encoder**. This notebook inspects its
    representations; it does not load the trained ThermoMPNN-D prediction head,
    so it does not report a meaningful ddG prediction.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    repo_root = Path.cwd()
    if not (repo_root / "configs").is_dir():
        repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "configs").is_dir():
        raise FileNotFoundError("Open this notebook from within the repository.")

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import (
        MegaScaleDatasetv2,
        tied_featurize_mut,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.model.modules import (
        get_protein_mpnn,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.train_thermompnn import (
        parse_cfg,
    )

    return (
        MegaScaleDatasetv2,
        OmegaConf,
        Path,
        alt,
        get_protein_mpnn,
        mo,
        parse_cfg,
        pd,
        repo_root,
        tied_featurize_mut,
        torch,
    )


@app.cell
def _(MegaScaleDatasetv2, OmegaConf, Path, parse_cfg, repo_root):
    paths_cfg = OmegaConf.load(repo_root / "configs/thermompnn_paths.example.yaml")
    double_cfg = OmegaConf.load(repo_root / "configs/thermompnn_train_epistatic.yaml")
    cfg = parse_cfg(OmegaConf.merge(paths_cfg, double_cfg))

    # Resolve configured data and checkpoint paths from the repository root.
    for key in ("megascale_splits", "megascale_pdbs", "megascale_csv"):
        path = Path(cfg.data_loc[key])
        if not path.is_absolute():
            cfg.data_loc[key] = str(repo_root / path)
    checkpoint_dir = Path(cfg.platform.thermompnn_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    cfg.platform.thermompnn_dir = str(checkpoint_dir)

    if cfg.model.aggregation != "siamese" or "double" not in cfg.data.mut_types:
        raise ValueError("Expected the epistatic double-mutant training config.")

    dataset = MegaScaleDatasetv2(cfg, split="val")
    if not len(dataset):
        raise ValueError("The MegaScale validation dataset has no usable examples.")

    checkpoint_files = sorted(checkpoint_dir.glob("v_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No ProteinMPNN checkpoints found in {checkpoint_dir}. "
            "Download the .pt files into checkpoints/proteinmpnn."
        )
    checkpoint_names = [path.name for path in checkpoint_files]
    default_checkpoint = (
        "v_48_020.pt" if "v_48_020.pt" in checkpoint_names else checkpoint_names[0]
    )
    return cfg, checkpoint_names, dataset, default_checkpoint


@app.cell(hide_code=True)
def _(dataset, mo):
    mo.md(f"""
    **Validation examples available:** {len(dataset):,}. Select a row below.
    The table is filtered to direct MegaScale double-mutant examples and sorted
    by protein length by the dataset implementation.
    """)
    return


@app.cell
def _(dataset):
    dataset.df[["WT_name", "mut_type", "ddG_ML", "aa_seq"]].head(12)
    return


@app.cell(hide_code=True)
def _(checkpoint_names, dataset, default_checkpoint, mo, torch):
    index_picker = mo.ui.number(
        start=0,
        stop=len(dataset) - 1,
        value=0,
        step=1,
        label="MegaScale validation row",
    )
    checkpoint_picker = mo.ui.dropdown(
        options=checkpoint_names,
        value=default_checkpoint,
        label="Frozen ProteinMPNN checkpoint",
    )
    available_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device_picker = mo.ui.dropdown(
        options=available_devices,
        value=available_devices[-1],
        label="Device",
    )
    mo.hstack([index_picker, checkpoint_picker, device_picker], justify="start")
    return checkpoint_picker, device_picker, index_picker


@app.cell
def _(dataset, index_picker):
    selected_index = int(index_picker.value)
    row = dataset.df.iloc[selected_index]
    sample = dataset[selected_index]
    mutation = sample["mutation"]
    return mutation, row, sample, selected_index


@app.cell
def _(sample):
    sample
    return


@app.cell(hide_code=True)
def _(mo, mutation, row, selected_index):
    mutation_lines = "\n".join(
        f"- `{wt}{position + 1}{mutant}`: {wt} → {mutant} at sequence position "
        f"{position + 1} (zero-based index {position})"
        for position, wt, mutant in zip(
            mutation.position, mutation.wildtype, mutation.mutation
        )
    )
    mo.md(f"""
    ### Selected example

    - **CSV row:** {selected_index}
    - **Protein:** `{row.WT_name}`
    - **MegaScale label:** `{row.mut_type}`
    - **Target passed to the model:** `{mutation.ddG:.4g}`

    {mutation_lines}

    The structure and sequence are the wild-type protein. The model receives
    the two mutation positions and residue identities as separate tensors.
    """)
    return


@app.cell
def _(sample, tied_featurize_mut):
    batch = tied_featurize_mut([sample], side_chains=False)

    return (batch,)


@app.cell
def _(batch):
    batch # tuple of length 12
    return


@app.cell
def _(batch):
    if batch is None:
        raise ValueError("Could not featurize this MegaScale example.")
    (
        X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        residue_idx,
        mut_positions,
        mut_wildtype_AAs,
        mut_mutant_AAs,
        mut_ddGs,
        atom_mask,
    ) = batch
    return (
        S,
        X,
        atom_mask,
        chain_M,
        chain_encoding_all,
        lengths,
        mask,
        mut_ddGs,
        mut_mutant_AAs,
        mut_positions,
        mut_wildtype_AAs,
        residue_idx,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `B` = batch size
    - `L` = sequence length
    - `L_max` = longest sequence in batch

    |  Tuple Index | Returned value       | Batch shape        | What it contains                                                                                                                                                          |
    | -: | -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    |  1 | `X`                  | `[B, L_max, 4, 3]` | Backbone atom coordinates for all proteins in the batch, padded to the longest sequence. With `side_chains=False`, each residue contributes 4 atoms: `N`, `CA`, `C`, `O`. |
    |  2 | `S`                  | `[B, L_max]`       | Amino-acid sequence IDs for each residue. Each entry is an index into the amino-acid alphabet; padded positions are zero-filled.                                          |
    |  3 | `mask`               | `[B, L_max]`       | Valid-residue mask: `1` for real residues and `0` for padded residues.                                                                                                    |
    |  4 | `lengths`            | `[B]`              | Original sequence length of each protein before padding.                                                                                                                  |
    |  5 | `chain_M`            | `[B, L_max]`       | Chain mask used by the model to distinguish masked vs. visible chains.                                                                                                    |
    |  6 | `chain_encoding_all` | `[B, L_max]`       | Chain ID for each residue. Residues from the same chain share the same value.                                                                                             |
    |  7 | `residue_idx`        | `[B, L_max]`       | Residue indices used internally by the model. Padded positions are set to `-100`.                                                                                         |
    |  8 | `MUT_POS`            | `[B, N_MUT]`       | Mutation positions for each example. `N_MUT` is the maximum number of mutations in the batch; shorter examples are zero-padded.                                           |
    |  9 | `MUT_WT_AA`          | `[B, N_MUT]`       | Wild-type amino-acid indices at each mutation site.                                                                                                                       |
    | 10 | `MUT_MUT_AA`         | `[B, N_MUT]`       | Mutant amino-acid indices at each mutation site.                                                                                                                          |
    | 11 | `MUT_DDG`            | `[B, 1]`           | Measured target ddG value for each example.                                                                                                                               |
    | 12 | `atom_mask`          | `[B, L_max, 4]`    | Missing-atom mask indicating where coordinates are missing or invalid. It complements `mask` by flagging missing atoms within residues.                                   |


    The “chains” are just different polypeptide chains in the structure:

    - a PDB can contain chain A, B, C, etc.
    - each chain has its own sequence and coordinates
    - this code stores them as things like:
        -  `seq_chain_A`
        -  `coords_chain_A`
        -  `seq_chain_B`
        -  `coords_chain_B`

    Inside  tied_featurize_mut , it does this:

    - reads all chain keys in the sample
    - separates them into:
        -  `masked_chains` : chains the model is asked to predict/mask
        -  `visible_chains` : chains it keeps visible as context
        - concatenates all selected chains into one long residue list
        - stores per-residue chain identity in  `chain_encoding_all`
        - stores chain-aware residue offsets in  `residue_idx`
        - uses  `chain_M`  as the mask telling which residues belong to predicted vs visible chains

    So the chain dimension is mostly there to preserve “which residues belong to which chain” while flattening everything into one batch tensor.

    - In the MegaScale dataset code, most examples are single-chain structures and the code explicitly uses:
    -  `chain = 'A' `
    -  `self.pdb_data[...]`
    - That means for the standard MegaScale setup, the structure is usually just chain A, so the chain concept is mostly bookkeeping rather than a major modeling feature.
    - It only becomes important when a protein complex or multi-chain structure is used, or when a  chain_dict  is supplied to mask some chains and keep others visible.

    In other words:

    - MegaScale usually = one protein, one relevant chain
    - multi-chain structures = chain labels matter for structure parsing, masking, and model input formatting

    This is why  `chain_encoding_all`  and  `residue_idx`  exist: they keep chain boundaries intact even after all residues are assembled into a single padded  `[B, L_max, ...`]  tensor.
    """)
    return


@app.cell
def _(X):
    X.shape # 1 x L x 4 x 3
    return


@app.cell
def _(S):
    S
    return


@app.cell
def _(S):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    sequence = "".join(alphabet[token] for token in S[0].tolist())
    sequence
    return alphabet, sequence


@app.cell
def _(mask):
    mask
    return


@app.cell
def _(lengths):
    lengths
    return


@app.cell
def _(chain_encoding_all):
    chain_encoding_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Inputs passed into ProteinMPNN

    `X` contains N, CA, C, and O coordinates. `S` is the wild-type sequence
    encoded as integer tokens. The masks and residue/chain indices tell the
    encoder which positions and neighbors are valid. Mutation positions and
    amino-acid IDs are retained for selecting mutation-site representations
    after the encoder runs.
    """)
    return


@app.cell
def _(
    S,
    X,
    atom_mask,
    chain_M,
    chain_encoding_all,
    mask,
    mut_ddGs,
    mut_mutant_AAs,
    mut_positions,
    mut_wildtype_AAs,
    pd,
    residue_idx,
):
    tensor_rows = []
    tensors = {
        "X: backbone coordinates": X,
        "S: wild-type sequence tokens": S,
        "mask: valid structured residues": mask,
        "chain_M: ProteinMPNN chain mask": chain_M,
        "chain_encoding_all: chain IDs": chain_encoding_all,
        "residue_idx: residue numbering": residue_idx,
        "mut_positions: mutation sites": mut_positions,
        "mut_wildtype_AAs: wild-type IDs": mut_wildtype_AAs,
        "mut_mutant_AAs: mutant IDs": mut_mutant_AAs,
        "mut_ddGs: regression target": mut_ddGs,
        "atom_mask": atom_mask,
    }
    for _name, _tensor in tensors.items():
        row_data = {
            "input": _name,
            "shape": str(tuple(_tensor.shape)),
            "dtype": str(_tensor.dtype),
        }
        tensor_rows.append(row_data)
    input_summary = pd.DataFrame(tensor_rows).set_index("input")
    input_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Instantiating the model
    """)
    return


@app.cell
def _(
    OmegaConf,
    cfg,
    checkpoint_picker,
    device_picker,
    get_protein_mpnn,
    torch,
):
    active_device = torch.device(device_picker.value)
    encoder_cfg = OmegaConf.merge(
        cfg, {"model": {"version": checkpoint_picker.value}}
    )
    protein_mpnn = get_protein_mpnn(encoder_cfg).to(active_device).eval()

    if any(parameter.requires_grad for parameter in protein_mpnn.parameters()):
        raise RuntimeError("Expected the configured ProteinMPNN to be frozen.")
    return active_device, protein_mpnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Running the model and extracting intermediate results

    Here we run the model and inspect the outputs, and we also use a PyTorch hook to extract the initial edges and edge features for the protein.
    """)
    return


@app.cell
def _(
    S,
    X,
    active_device,
    chain_M,
    chain_encoding_all,
    mask,
    protein_mpnn,
    residue_idx,
    torch,
):
    model_inputs = (
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
    )
    model_inputs = tuple(tensor.to(active_device) for tensor in model_inputs)
    feature_capture = {}

    def capture_protein_features(_module, _inputs, output):
        feature_capture["E"], feature_capture["E_idx"] = output

    feature_hook = protein_mpnn.features.register_forward_hook(
        capture_protein_features
    )
    # get outputs of the model
    try:
        with torch.inference_mode():
            hidden_layers, wt_embed, log_probs, h_E = protein_mpnn(*model_inputs)
    finally:
        feature_hook.remove()

    if "E" not in feature_capture:
        raise RuntimeError("ProteinFeatures hook did not capture graph features.")

    E = feature_capture["E"]
    E_idx = feature_capture["E_idx"]
    return E_idx, h_E, hidden_layers, log_probs, wt_embed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. ProteinFeatures builds the structural graph

    For each residue, `ProteinFeatures` finds nearby residues using CA distances.
    It describes each residue-to-neighbor connection with backbone-atom
    distances and relative sequence position, then embeds those into `E`.
    `E_idx` records which residue is each neighbor. ProteinMPNN uses `E` and
    `E_idx` as graph edges for its encoder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. ProteinMPNN propagates information over the graph

    - `h_E` is the learned edge representation after ProteinMPNN's encoder layers.
    - `hidden_layers` contains the decoder's residue representations; they are
    returned in reverse order, so this Siamese model takes the first two for
    - `num_final_layers: 2`.
    - `wt_embed` is the learned embedding of each
    wild-type amino acid.
    - `log_probs` are ProteinMPNN's amino-acid log
    probabilities; ThermoMPNN-D does not use those as its ddG prediction here.
    """)
    return


@app.cell(hide_code=True)
def _(h_E, hidden_layers, log_probs, pd, wt_embed):
    mpnn_outputs = [
        ("h_E: learned graph-edge features", h_E),
        ("wt_embed: sequence embeddings", wt_embed),
        ("log_probs: amino-acid log probabilities", log_probs),
    ]
    mpnn_outputs.extend(
        (f"hidden_layers[{index}]: decoder representation", tensor)
        for index, tensor in enumerate(hidden_layers)
    )
    mpnn_rows = []
    for _name, _tensor in mpnn_outputs:
        _values = _tensor.detach().float()
        mpnn_rows.append(
            {
                "output": _name,
                "shape": str(tuple(_tensor.shape)),
                "mean": float(_values.mean()),
                "std": float(_values.std(unbiased=False)),
                "min": float(_values.min()),
                "max": float(_values.max()),
            }
        )
    mpnn_summary = pd.DataFrame(mpnn_rows).set_index("output")
    mpnn_summary
    return


@app.cell(hide_code=True)
def _(hidden_layers, mo):
    selected_hidden = hidden_layers[:2]
    mo.md(
        "**Representations used by the double-mutant head:** "
        + ", ".join(
            f"`hidden_layers[{index}]` with shape `{tuple(tensor.shape)}`"
            for index, tensor in enumerate(selected_hidden)
        )
        + ". These are the last two decoder layers because ProteinMPNN returns "
        "the decoder layers in reverse order."
    )
    return (selected_hidden,)


@app.cell
def _(
    E_idx,
    alphabet,
    h_E,
    mut_mutant_AAs,
    mutation,
    pd,
    protein_mpnn,
    selected_hidden,
    sequence,
    torch,
    wt_embed,
):
    site_rows = []
    site_vectors = []
    for site_number, _position in enumerate(mutation.position):
        structural_vector = torch.cat(
            [layer[0, _position, :] for layer in selected_hidden], dim=-1
        )
        wildtype_vector = wt_embed[0, _position, :]
        mutant_id = mut_mutant_AAs[0, site_number].to(wt_embed.device)
        mutant_vector = protein_mpnn.W_s(mutant_id)
        sequence_difference = wildtype_vector - mutant_vector

        other_site = mutation.position[1 - site_number]
        neighbor_positions = E_idx[0, _position]
        matching_neighbors = torch.where(neighbor_positions == other_site)[0]
        if matching_neighbors.numel():
            edge_vector = h_E[0, _position, matching_neighbors[0], :]
            edge_status = f"present; other mutation is neighbor rank {int(matching_neighbors[0]) + 1}"
        else:
            edge_vector = torch.zeros_like(h_E[0, _position, 0, :])
            edge_status = "not in the neighbor list; Siamese code uses zeros"

        combined = torch.cat([structural_vector, sequence_difference, edge_vector])
        site_vectors.append(combined)

        site_rows.append(
            {
                "site": f"{sequence[_position]}{_position + 1}{alphabet[int(mutant_id)]}",
                "zero_based_position": _position,
                "wildtype_embedding_shape": str(tuple(wildtype_vector.shape)),
                "mutant_embedding_shape": str(tuple(mutant_vector.shape)),
                "structure_features_shape": str(tuple(structural_vector.shape)),
                "WT_minus_mutant_embedding_shape": str(tuple(sequence_difference.shape)),
                "pair_edge": edge_status,
                "edge_features_shape": str(tuple(edge_vector.shape)),
                "concatenated_site_features_shape": str(tuple(combined.shape)),
            }
        )
    site_feature_table = pd.DataFrame(site_rows).set_index("site")
    site_vectors = torch.stack(site_vectors)
    return site_feature_table, site_vectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. What the Siamese head extracts at each mutation

    For each site, `TransferModelv2Siamese` concatenates:

    1. The two selected ProteinMPNN decoder representations (256 values).
    2. The wild-type sequence embedding minus the replacement-amino-acid
       embedding (128 values).
    3. The learned edge representation connecting the two mutation sites
       (128 values), or zeros if the other site is absent from the neighbor list.

    With this config that makes **512 values per mutation site**. A learned
    projection reduces each site vector before the prediction head combines
    the sites in AB and BA order. The table below shows the extracted shapes
    and whether the inter-mutation edge was found.
    """)
    return


@app.cell
def _(site_feature_table):
    site_feature_table
    return


@app.cell
def _(alt, mo, pd, site_vectors):
    vector_values = site_vectors.detach().cpu().numpy()
    feature_plot_data = pd.DataFrame(
        [
            {
                "mutation_site": f"site {site_number + 1}",
                "feature_index": feature_index,
                "value": float(value),
                "feature_group": (
                    "decoder structure"
                    if feature_index < 256
                    else "WT - mutant sequence"
                    if feature_index < 384
                    else "inter-mutation edge"
                ),
            }
            for site_number, row_values in enumerate(vector_values)
            for feature_index, value in enumerate(row_values)
        ]
    )
    feature_heatmap = (
        alt.Chart(feature_plot_data)
        .mark_rect()
        .encode(
            x=alt.X("feature_index:O", title="Concatenated feature index"),
            y=alt.Y("mutation_site:N", title=None),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
            tooltip=["mutation_site", "feature_index", "feature_group", "value"],
        )
        .properties(
            title="Extracted 512-dimensional site vectors",
            width=700,
            height=100,
        )
    )
    mo.ui.altair_chart(feature_heatmap)
    return


@app.cell(hide_code=True)
def _(active_device, checkpoint_picker, mo, protein_mpnn):
    frozen_parameter_count = sum(
        parameter.numel() for parameter in protein_mpnn.parameters()
    )
    mo.md(
        f"**Encoder:** `{checkpoint_picker.value}` · **device:** `{active_device}` · "
        f"**frozen parameters:** {frozen_parameter_count:,}"
    )
    return


if __name__ == "__main__":
    app.run()
