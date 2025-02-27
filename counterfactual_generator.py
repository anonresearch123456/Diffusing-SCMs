# create_counterfactuals.py

import os
import h5py
import hydra
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from typing import List, Tuple, Any, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import itertools

# --------------------------------------------------------------------------------------
# Reuse your pipeline code where appropriate
# --------------------------------------------------------------------------------------
from src.models.lightningmodules.diffusion import LDMModule, TabularEncoder
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDIMScheduler
from src.models.lightningmodules.predictmeta import MetadataPredictionAll
from src.data.ukbdata import (
    standardize_age,
    get_context_vector,
    one_hot_encode_site,
    SITE_MAP
)
from src.data.utils.splitter import DataSplitter


# --------------------------------------------------------------------------------------
# Utility to replicate your forward pass with padding logic
# --------------------------------------------------------------------------------------
def forward(model: torch.nn.Module, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, padding: bool):
    x = F.pad(x, (1, 1, 2, 2, 2, 2), mode='reflect') if padding else x
    out = model(x, timesteps=timesteps, context=context)
    out = out[:, :, 2:-2, 2:-2, 1:-1] if padding else out
    return out


# --------------------------------------------------------------------------------------
# HDF5 storing utilities, adapted to store rows as pd.Series in the same order
# --------------------------------------------------------------------------------------
def tabular_to_h5_dataset(row_series: pd.Series, group: h5py.Group) -> None:
    """
    Store the entire row (in the same column order as the original CSV) into the 'tabular' dataset.
    We convert to float32. This means all columns from the original CSV are stored in HDF5 
    (except any extra columns you might have added, which we won't pass in row_series).

    IMPORTANT: row_series must already be in the exact order you want stored.
    """
    # Convert to float32 for consistency
    data_array = row_series.values.astype(np.float32)
    group.create_dataset(
        "tabular",
        data=data_array,
        compression="gzip"
    )

def image_to_h5_dataset(img: np.ndarray, group: h5py.Group) -> None:
    """
    Store the generated image volume as an uint8 array [0..255].
    """
    group.create_dataset(
        "mri",
        data=img,
        compression="gzip"
    )

def noise_to_h5_dataset(noise: np.ndarray, group: h5py.Group) -> None:
    """
    Store the parent's (initial) noise, unaltered.
    """
    group.create_dataset(
        "initial_noise",
        data=noise,
        compression="gzip"
    )


def sample_counterfactuals(
    row: pd.Series,
    columns_to_vary: List[str],
    age_range: Tuple[float, float],
    n_samples: int
) -> List[pd.Series]:
    """
    Given a single row (pd.Series) from the original CSV, produce a list of new rows (pd.Series),
    each corresponding to a counterfactual for the specified columns by sampling randomly.

    For categorical/binary columns:
        - "Sex_true": sample uniformly from [0, 1].
        - "Site_true": sample uniformly from [11025, 11026, 11027].
    For the continuous "Age_true":
        - Sample uniformly from the given age_range, rounding to an integer.
    
    The function repeats the sampling process until n_samples counterfactual rows are generated.
    Repeated values and even the already realized value are allowed.
    
    The returned row objects share the exact same index (column labels) as 'row'.
    """
    # Define the full set of options for categorical variables.
    categorical_options = {
        "Sex_true": [0, 1],
        "Site_true": [11025, 11026, 11027]
    }
    
    cf_rows = []
    for _ in range(n_samples):
        cf_row = row.copy()
        for col in columns_to_vary:
            if col == "Age_true":
                low, high = age_range
                # Sample uniformly and round to an integer.
                sampled_age = np.random.uniform(low, high)
                cf_row[col] = int(round(sampled_age))
            elif col in categorical_options:
                # Sample randomly from the specified options.
                cf_row[col] = np.random.choice(categorical_options[col])
            else:
                raise ValueError(f"Column {col} not recognized or not supported for variation.")
        cf_rows.append(cf_row)
    
    return cf_rows


# --------------------------------------------------------------------------------------
# Generate a single counterfactual image from the parent's noise + the modified row
# --------------------------------------------------------------------------------------
def generate_counterfactual_image(
    encoder: AutoencoderKL,
    diffusion_model: DiffusionModelUNet,
    scheduler: DDIMScheduler,
    classifier: MetadataPredictionAll,
    tabular_encoder: TabularEncoder,
    cf_row: pd.Series,
    noise: torch.Tensor,
    device: str,
    ldm_conf: DictConfig,
    guidance_scale: float,
    age_scale: float,
    sex_scale: float,
    site_scale: float,
) -> np.ndarray:
    """
    Exactly the same logic as your pipeline's generation, but re-using `noise` (the parent's noise)
    and the updated tabular context from `cf_row`.
    """
    cross_cond: bool = ldm_conf.cross_cond
    concat_cond: bool = ldm_conf.concat_cond
    in_channels_latent_space: int = (
        ldm_conf.diffusion.in_channels - 5 if concat_cond else ldm_conf.diffusion.in_channels
    )
    scale_factor: float = ldm_conf.scale_factor
    padding: bool = ldm_conf.padding

    # Move noise to device and clone so we don't modify the parent's copy
    noise = noise.to(device).clone()

    # Prepare tabular context from the CF row
    age_std_tensor = standardize_age(torch.as_tensor(cf_row["Age_true"], dtype=torch.float32)).to(device)
    sex_tensor = torch.as_tensor(cf_row["Sex_true"], dtype=torch.float32).to(device)
    site_idx = SITE_MAP[cf_row["Site_true"]]
    site_ohe = one_hot_encode_site(torch.as_tensor(site_idx, dtype=torch.long)).to(device)
    tabular_context = get_context_vector(age_std_tensor, sex_tensor, site_ohe).unsqueeze(0).reshape(1, 1, -1).to(device)

    # Expand tabular context for concat if needed
    tabular_context_v = tabular_context.view(tabular_context.size(0), tabular_context.size(2), 1, 1, 1)
    tabular_context_v = tabular_context_v.expand(-1, -1, 12, 12, 14)

    # Optional cross-attention embeddings
    context_embeddings = tabular_encoder(tabular_context) if cross_cond else None

    # Reverse diffusion steps
    timesteps_list = scheduler.timesteps
    for t in tqdm(timesteps_list, leave=False):
        noise_input = torch.cat([noise, tabular_context_v], dim=1) if concat_cond else noise
        timesteps_tensor = torch.Tensor((t,)).to(device).long()

        with torch.autocast(device_type=device, dtype=torch.float16):
            model_output = forward(diffusion_model, noise_input, timesteps_tensor, context_embeddings, padding)
            _, pred_original_sample = scheduler.step(model_output, t, noise)

            # Classifier guidance
            pred_original_sample.requires_grad_(True)
            pred_original_sample_decoded = encoder.decode(pred_original_sample / scale_factor)
            pred_original_sample_decoded = torch.clamp(pred_original_sample_decoded, 0, 1)

            age_loss, sex_loss, site_loss = classifier.get_prediction_losses(
                pred_original_sample_decoded,
                age_std_tensor,
                sex_tensor,
                torch.as_tensor(site_idx, dtype=torch.long, device=device)
            )
            guidance_loss = age_scale * age_loss + sex_scale * sex_loss + site_scale * site_loss

            grad = torch.autograd.grad(
                guidance_loss, pred_original_sample, retain_graph=False, create_graph=False
            )[0]

            model_output += guidance_scale * grad

        noise, _ = scheduler.step(model_output, t, noise)

    # Decode final sample
    with torch.no_grad():
        sample = encoder.decode(noise / scale_factor)
        sample = np.clip(sample.cpu().numpy(), 0, 1)
        sample = (sample * 255).squeeze()
        sample = sample.astype(np.uint8)

    return sample


# --------------------------------------------------------------------------------------
# Main Hydra function to generate counterfactuals
# --------------------------------------------------------------------------------------
@hydra.main(config_path="config/counterfactual_gen", config_name="diffusion_counterfactual.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """
    Example Hydra entry point for creating counterfactual images:

      - path_to_original_h5: the existing synthetic data .h5
      - path_to_original_csv: the corresponding CSV
      - columns_to_vary: e.g. ["Sex_true", "Site_true"]
      - age_range: e.g. [40, 80]
      - n_age_samples: how many to sample for Age_true
      - proportion_of_rows: fraction of rows from the original CSV to produce CF for
      - new_out_dir, new_out_h5_name: location of new CF data
      - guidance_scale, age_scale, sex_scale, site_scale, ...
      - scheduler.timesteps, ...
    """

    # 1) Device, set up
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) Load LDM modules (diffusion, autoencoder, tabular encoder, classifier)
    config_ldm = OmegaConf.load(cfg.diffusion.cfg_path)

    if "cfg_path" in cfg.encoder and cfg.encoder.cfg_path:
        config_ldm.autoencoder.cfg_path = cfg.encoder.cfg_path
    if "checkpoint_path" in cfg.encoder and cfg.encoder.checkpoint_path:
        config_ldm.autoencoder.checkpoint_path = cfg.encoder.checkpoint_path

    encoder, diffusion_model, scheduler, tabular_encoder = LDMModule.get_diffusion_for_inference(
        cfg=config_ldm,
        checkpoint_path=cfg.diffusion.checkpoint_path
    )
    classifier = MetadataPredictionAll.load_from_checkpoint(cfg.classifier.checkpoint_path)

    scheduler.set_timesteps(cfg.scheduler.timesteps)

    encoder, diffusion_model, tabular_encoder, classifier = (
        encoder.to(device),
        diffusion_model.to(device),
        tabular_encoder.to(device),
        classifier.to(device)
    )
    encoder.eval()
    diffusion_model.eval()
    tabular_encoder.eval()
    classifier.eval()
    encoder.requires_grad_(False)
    diffusion_model.requires_grad_(False)
    tabular_encoder.requires_grad_(False)
    classifier.requires_grad_(False)

    path_to_data = Path(cfg.path_to_data)

    split_file = path_to_data / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {split_file} not found. Please generate a split first.")

    # read json split file
    with open(split_file, "r") as f:
        split_dict = json.load(f)

    test_eids = split_dict["test"]

    # subset df only on test eids
    df = pd.read_csv(path_to_data / "diffusion_data.csv")
    df = df[df["eid"].isin(test_eids)].reset_index(drop=True)
    n_samples = cfg.n_samples
    df_subset = df.sample(n_samples, replace=False).reset_index(drop=True)

    # 4) Open the original H5 in read mode
    original_h5 = path_to_data / "diffusion_data.h5"
    fin = h5py.File(original_h5, "r")

    # 5) Prepare new output H5
    out_dir = Path(cfg.new_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_h5_path = out_dir / cfg.new_out_h5_name
    if new_h5_path.exists():
        raise FileExistsError(f"{new_h5_path} already exists; please remove or specify a different path.")

    fout = h5py.File(new_h5_path, "w")

    # We'll collect new CF rows in a list, then write a new CSV at the end
    cf_rows = []
    new_eid_counter = 1  # EIDs for newly created CF rows

    columns = df.columns.tolist()  # The original columns in correct order
    columns_to_vary = cfg.columns_to_vary  # e.g. ["Sex_true", "Site_true"]
    age_range = tuple(cfg.age_range) if "age_range" in cfg else (50, 80)
    n_cfs = cfg.n_cfs if "n_cfs" in cfg else 4

    # We want to add one extra column "orig_eid" at the end for CSV
    # but not store it in the H5's 'tabular' dataset
    # So the final CSV columns will be original columns + ["orig_eid"]
    final_csv_columns = columns + ["orig_eid"]

    pbar = tqdm(total=len(df_subset), desc="Creating CFs")
    for _, row in df_subset.iterrows():
        # row is a pd.Series with the same columns as 'df'
        orig_eid = str(row["eid"])
        if orig_eid not in fin:
            print(f"WARNING: EID {orig_eid} not found in the original H5. Skipping.")
            pbar.update()
            continue

        # Read parent's noise
        parent_noise_np = fin[orig_eid]["initial_noise"][:]
        parent_noise_torch = torch.from_numpy(parent_noise_np).float().to(device)

        # Enumerate all CF combos
        cf_variants = sample_counterfactuals(row, columns_to_vary, age_range, n_cfs)

        # For each variant, create the new image
        for cf_row in cf_variants:
            # Force a copy so we can set a new EID
            cf_row_copy = cf_row.copy()

            # 1) Overwrite "eid" with a new CF EID
            cf_row_copy["eid"] = new_eid_counter

            # 2) Actually generate the new image
            new_mri = generate_counterfactual_image(
                encoder=encoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                classifier=classifier,
                tabular_encoder=tabular_encoder,
                cf_row=cf_row_copy,
                noise=parent_noise_torch,
                device=device,
                ldm_conf=config_ldm.ldm,
                guidance_scale=cfg.guidance_scale,
                age_scale=cfg.age_scale,
                sex_scale=cfg.sex_scale,
                site_scale=cfg.site_scale,
            )

            # 3) Create H5 group
            grp = fout.create_group(str(new_eid_counter))

            cf_row_copy["Age"] = cf_row_copy["Age_true"]
            cf_row_copy["Sex"] = cf_row_copy["Sex_true"]
            cf_row_copy["Site"] = cf_row_copy["Site_true"]

            # 4) Write tabular data (original columns only) 
            #    in the exact order, as a pd.Series
            tabular_to_h5_dataset(cf_row_copy[columns], grp)

            # 5) Write image
            image_to_h5_dataset(new_mri, grp)

            # 6) Write parent's noise as is
            noise_to_h5_dataset(parent_noise_np, grp)

            # 7) Build row for CSV output: 
            #    all original columns + "orig_eid" at the end
            out_row_series = cf_row_copy[columns].copy()
            # Append the parent's EID as a new field
            out_row_series["orig_eid"] = orig_eid

            # Convert to a dict or keep as series, then store in a list
            cf_rows.append(out_row_series)

            new_eid_counter += 1

        pbar.update()

    pbar.close()
    fin.close()
    fout.close()

    # -----------------------------------------------------------------------------------
    # Build new CSV with final columns = original columns + ["orig_eid"]
    # -----------------------------------------------------------------------------------
    cf_df = pd.DataFrame(cf_rows, columns=final_csv_columns)
    new_csv_path = new_h5_path.with_suffix(".csv")
    cf_df.to_csv(new_csv_path, index=False)

    # Optionally, you can create a new train/val/test split for your new CF dataset
    splitter = DataSplitter(new_h5_path.parent, cf_df)
    splitter.generate_split("split_counterfactual", test_ratio=cfg.test_ratio, val_ratio=cfg.val_ratio)

    print(f"Counterfactual generation complete.\nNew H5:  {new_h5_path}\nNew CSV: {new_csv_path}")


if __name__ == "__main__":
    main()
