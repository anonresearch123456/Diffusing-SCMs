import os
import h5py
import hydra
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

from typing import Any, Tuple
from tqdm import tqdm
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDIMScheduler
from omegaconf import DictConfig, OmegaConf
from src.models.lightningmodules.predictmeta import MetadataPredictionAll
from src.data.utils.splitter import DataSplitter


from src.data.ukbdata import standardize_age, get_context_vector, one_hot_encode_site, SITE_MAP
from src.models.lightningmodules.diffusion import LDMModule, TabularEncoder
import json
import random
from pathlib import Path


def sample_like_orig(original_csv_path: str, n_samples: int) -> pd.DataFrame:
    data = pd.read_csv(original_csv_path)

    # sample with or without replacement
    synthetic_data = data.sample(n=n_samples, replace=False).reset_index(drop=True)

    # Add a new 'eid' column with unique sequential IDs
    synthetic_data['eid'] = range(1, len(synthetic_data) + 1)

    return synthetic_data


def sample_from_formulas(scm: dict, n_samples: int) -> pd.DataFrame:
    # not implemented here
    pass


def forward(model: torch.nn.Module, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, padding: bool):
    x = F.pad(x, (1, 1, 2, 2, 2, 2), mode='reflect') if padding else x
    out = model(x, timesteps=timesteps, context=context)
    # remove padding
    out = out[:, :, 2:-2, 2:-2, 1:-1] if padding else out
    return out


def tabular_to_h5_dataset(row: Tuple[Any], group: h5py.Group) -> None:
    features = row[1:]  # skip index value
    group.create_dataset(
        "tabular",
        data=np.array(features, dtype=np.float32),
        compression="gzip"
    )


def image_to_h5_dataset(img: np.ndarray, group: h5py.Group) -> None:
    group.create_dataset(
        "mri",
        data=img,
        compression="gzip"
    )


### NEW / MODIFIED ###
def noise_to_h5_dataset(noise: torch.Tensor, group: h5py.Group) -> None:
    """Store the initial noise in the H5 file."""
    noise_np = noise.detach().cpu().numpy()
    group.create_dataset(
        "initial_noise",
        data=noise_np,
        compression="gzip"
    )


def generate_and_save_MRIs(
    encoder: AutoencoderKL,
    diffusion_model: DiffusionModelUNet,
    scheduler: DDIMScheduler,
    classifier: MetadataPredictionAll,
    tabular_encoder: TabularEncoder,
    data_frame: pd.DataFrame,
    ldm_conf: DictConfig,
    guidance_scale: float,
    age_scale: float,
    sex_scale: float,
    site_scale: float,
    out_h5_name: str
) -> None:

    encoder.eval()
    diffusion_model.eval()
    tabular_encoder.eval()
    classifier.eval()

    encoder.requires_grad_(False)
    diffusion_model.requires_grad_(False)
    tabular_encoder.requires_grad_(False)
    classifier.requires_grad_(False)

    scheduler.clip_sample = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    diffusion_model = diffusion_model.to(device)
    tabular_encoder = tabular_encoder.to(device)
    classifier = classifier.to(device)

    cross_cond: bool = ldm_conf.cross_cond
    concat_cond: bool = ldm_conf.concat_cond
    in_channels_latent_space: int = (
        ldm_conf.diffusion.in_channels - 5 if concat_cond else ldm_conf.diffusion.in_channels
    )
    scale_factor: float = ldm_conf.scale_factor
    padding: bool = ldm_conf.padding

    # assert that out_h5_name does not exist
    assert not os.path.exists(out_h5_name), f"{out_h5_name} already exists"

    with h5py.File(out_h5_name, mode="w") as fout:
        pbar = tqdm(total=data_frame.shape[0])
        for row in data_frame.itertuples():
            # Prepare tabular context
            age_std_tensor = standardize_age(torch.as_tensor(row.Age_true, dtype=torch.float32))
            site_tensor = one_hot_encode_site(torch.as_tensor(SITE_MAP[row.Site_true], dtype=torch.long))
            sex_tensor = torch.as_tensor(row.Sex_true, dtype=torch.float32)
            tabular_context = get_context_vector(age_std_tensor, sex_tensor, site_tensor)
            tabular_context = tabular_context.unsqueeze(0).reshape(1, 1, -1).to(device)

            age_std_tensor = age_std_tensor.to(device)
            sex_tensor = sex_tensor.to(device)
            site_tensor = torch.as_tensor(SITE_MAP[row.Site_true], dtype=torch.long).to(device)

            # Create initial noise
            noise = torch.randn(1, in_channels_latent_space, 12, 12, 14).to(device)

            group = fout.create_group(str(row.eid))

            noise_to_h5_dataset(noise, group)

            # Expand tabular context if concat_cond
            tabular_context_v = tabular_context.view(tabular_context.size(0), tabular_context.size(2), 1, 1, 1)
            tabular_context_v = tabular_context_v.expand(-1, -1, 12, 12, 14)

            context_embeddings = tabular_encoder(tabular_context) if cross_cond else None

            progress_bar = tqdm(scheduler.timesteps, leave=False)
            for t in progress_bar:
                noise_input = torch.cat([noise, tabular_context_v], dim=1) if concat_cond else noise
                timesteps = torch.Tensor((t,)).to(noise.device).long()
                # classifier guidance with autocast
                with torch.autocast(device_type=device, dtype=torch.float16):
                    model_output = forward(diffusion_model, noise_input, timesteps, context_embeddings, padding)
                    _, pred_original_sample = scheduler.step(model_output, t, noise)

                    pred_original_sample.requires_grad_(True)
                    pred_original_sample_decoded = encoder.decode(pred_original_sample / scale_factor)
                    pred_original_sample_decoded = torch.clamp(pred_original_sample_decoded, 0, 1)

                    age_loss, sex_loss, site_loss = classifier.get_prediction_losses(
                        pred_original_sample_decoded, age_std_tensor, sex_tensor, site_tensor
                    )
                    guidance_loss = age_scale * age_loss + sex_scale * sex_loss + site_scale * site_loss

                    grad = torch.autograd.grad(
                        guidance_loss, pred_original_sample, retain_graph=False, create_graph=False
                    )[0]

                    model_output += guidance_scale * grad

                noise, _ = scheduler.step(model_output, t, noise)

            with torch.no_grad():
                sample = encoder.decode(noise / scale_factor)
                sample = np.clip(sample.cpu().numpy(), 0, 1)
                sample = (sample * 255).squeeze()
                sample = sample.astype(np.uint8)

            tabular_to_h5_dataset(row, group)
            image_to_h5_dataset(sample, group)
            ### NEW / MODIFIED ###
            # Store the initial noise in the same group

            pbar.update()
        pbar.close()

    # save csv file with the same name as the h5 file
    csv_name = out_h5_name.replace(".h5", ".csv")
    data_frame.to_csv(csv_name, index=False)


@hydra.main(config_path="config", config_name="diffusion_dataset_creator_SCM.yaml", version_base="1.3")
def main(config: DictConfig):
    # 1) Load your LDM modules
    config_path_ldm = config.diffusion.cfg_path
    checkpoint_path_ldm = config.diffusion.checkpoint_path

    config_ldm = OmegaConf.load(config_path_ldm)
    # Possibly override the autoencoder config if needed
    config_ldm.autoencoder.cfg_path = config.encoder.cfg_path if "cfg_path" in config.encoder and config.encoder.cfg_path else config_ldm.autoencoder.cfg_path
    config_ldm.autoencoder.checkpoint_path = config.encoder.checkpoint_path if "checkpoint_path" in config.encoder and config.encoder.checkpoint_path else config_ldm.autoencoder.checkpoint_path

    encoder, diffusion_model, scheduler, tabular_encoder = LDMModule.get_diffusion_for_inference(
        cfg=config_ldm,
        checkpoint_path=checkpoint_path_ldm
    )
    classifier = MetadataPredictionAll.load_from_checkpoint(config.classifier.checkpoint_path)
    scheduler.set_timesteps(config.scheduler.timesteps)

    # 2) Decide how you sample (original or formulas)
    sample_type: str = config.sample_type
    original_csv_path: str = config.original_csv_path
    n_samples: int = config.n_samples
    scm: dict = config.SCM

    if sample_type == "original":
        data_frame = sample_like_orig(original_csv_path=original_csv_path, n_samples=n_samples)
    elif sample_type == "formulas":
        data_frame = sample_from_formulas(scm=scm, n_samples=n_samples)
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    # 3) Generate and save
    out_h5_name = config.out_h5_name
    generate_and_save_MRIs(
        encoder=encoder,
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        classifier=classifier,
        tabular_encoder=tabular_encoder,
        data_frame=data_frame,
        ldm_conf=config_ldm.ldm,
        guidance_scale=config.guidance_scale,
        age_scale=config.age_scale,
        sex_scale=config.sex_scale,
        site_scale=config.site_scale,
        out_h5_name=out_h5_name
    )

    splitter = DataSplitter(Path(out_h5_name).parent, data_frame)  # or pass the final CSV
    splitter.generate_split("split", test_ratio=config.test_ratio, val_ratio=config.val_ratio)

    split_json_path = Path(out_h5_name).parent / "split.json"

    with open(split_json_path, "r") as f:
        split_dict = json.load(f)
    test_ids = split_dict["test"]


if __name__ == "__main__":
    main()
