import os
import sys
import logging
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

# Import your modules (adjust the imports if needed)
from src.data.ukbdata import UKBDataModule
from src.debiasing.models import AdversarialPredictor, cmmdRegularizedPredictor, MetaDataPrediction

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Base experiment directory
BASE_EXPERIMENT_DIR = "..."

# Define dataset paths (you can adjust or expand these as needed)
DATASET = [("diffusion_age_binary_balanced",
            ".../diffusion_data.h5",
            ".../split.json"),]

DATASET_CTF = [("confounding_experiment_ctf",
                ".../cf_data.h5",
                ".../split.json")]

CTF_CSV_PATH = ".../cf_data.csv"



def extract_hparams(config, method):
    """
    Extract key hyperparameters from the config depending on the method.
    For adversarial, we extract lambda_feature, lambda_protected, etc.
    For cmmd, we extract cmmd_lambda, inverted_bandwidth, etc.
    """
    hparams = {}
    if method == "adversarial":
        pred_cfg = config.get("predictor", {})
        hparams["lambda_feature"] = pred_cfg.get("lambda_feature")
        hparams["lambda_protected"] = pred_cfg.get("lambda_protected")
        hparams["protected_pred_head"] = pred_cfg.get("protected_pred_head")
        # Add any other adversarial-specific hyperparameters here
    elif method == "cmmd":
        pred_cfg = config.get("predictor", {})
        hparams["cmmd_lambda"] = pred_cfg.get("cmmd_lambda")
        hparams["inverted_bandwidth"] = pred_cfg.get("inverted_bandwidth")
        # Add any other cmmd-specific hyperparameters here
    else:
        # For baseline or other methods, add hyperparameters as needed
        hparams["note"] = "baseline/default hyperparams"
    return hparams


def run_test(model_triplet: tuple, device: str = "cuda:0", batch_size: int = 32):
    """
    Run the test pipeline for one model given (name, config_path, checkpoint_path).
    Returns a dict with the metrics.
    """
    model_name, config_path, ckpt_path = model_triplet

    LOG.info(f"Testing model: {model_name}")
    model_cfg = OmegaConf.load(config_path)

    # Load model based on model name
    if "adversarial" in model_name:
        model = AdversarialPredictor.load_from_checkpoint(ckpt_path)
    elif "cmmd" in model_name:
        model = cmmdRegularizedPredictor.load_from_checkpoint(ckpt_path)
    else:
        model = MetaDataPrediction.load_from_checkpoint(ckpt_path)

    model = model.to(device)
    model.eval()
    model.freeze()

    # Choose the dataset (here we first run the diffusion test set)
    dataset_name, dataset_path, split_path = DATASET[0]
    data_module = UKBDataModule(dataset_path, split_path, batch_size=batch_size, num_workers=4,
                                crop_images=False, has_bin_age=True)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # Run inference and collect metrics
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.test_step(batch, batch_idx)
    metrics = model.on_test_epoch_end()
    LOG.info(f"Metrics on diffusion dataset: {metrics}")

    # Now run on the counterfactual dataset and compute additional metrics
    dataset_name_ctf, dataset_path_ctf, split_path_ctf = DATASET_CTF[0]
    data_module = UKBDataModule(dataset_path_ctf, split_path_ctf, batch_size=batch_size, num_workers=4,
                                crop_images=False, has_bin_age=True, has_orig_eid=True)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    df = pd.DataFrame()
    # We assume the CSV with mapping for original eid is in a fixed location:
    dataframe = pd.read_csv(CTF_CSV_PATH)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing CTF {model_name}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        mri = batch["mri"]
        target = batch[model.target]
        eid = batch["eid"].cpu().detach().numpy().squeeze()

        # For each element in batch, get the original eid from the dataframe
        orig_eid = []
        for e in eid:
            orig_eid.append(dataframe[dataframe["eid"] == e]["orig_eid"].values[0])
        orig_eid = torch.tensor(orig_eid).cpu().detach().numpy().squeeze()

        y_hat = model.forward(mri).squeeze()
        y_pred = torch.sigmoid(y_hat)
        y_pred = (y_pred > 0.5).float()
        y_pred = y_pred.cpu().detach().numpy().squeeze()
        y_hat = y_hat.cpu().detach().numpy().squeeze()

        df = pd.concat([df, pd.DataFrame({"orig_eid": orig_eid, "y_hat": y_hat, "y_pred": y_pred})])
    
    # Compute variances as additional metrics
    df_v1 = df.groupby("orig_eid").agg({"y_pred": "std"}).reset_index()
    mean_var_pred = df_v1["y_pred"].mean()
    df_v2 = df.groupby("orig_eid").agg({"y_hat": "std"}).reset_index()
    mean_var_hat = df_v2["y_hat"].mean()
    LOG.info(f"CTF dataset mean variance (y_pred): {mean_var_pred}")
    LOG.info(f"CTF dataset mean variance (y_hat): {mean_var_hat}")

    # get bacc from metrics. note: can be stored as bacc or age_bin/bacc
    metrics["bacc"] = metrics.get("bacc", metrics.get("age_bin/bacc", None))
    metrics["roc_auc"] = metrics.get("roc_auc", metrics.get("age_bin/roc_auc", None))

    # Combine metrics in a dictionary to return
    results = {
        "model_name": model_name,
        "bacc": metrics["bacc"],
        "roc_auc": metrics["roc_auc"],
        "ctf_mean_var_pred": mean_var_pred,
        "ctf_mean_var_hat": mean_var_hat,
    }
    return results, model_cfg


def discover_runs(method_dir: Path):
    """
    Discover all experiment runs under a given method directory.
    Each run should have a .hydra/config.yaml and a chkpts directory with a checkpoint.
    Returns a list of tuples: (model_name, config_path, checkpoint_path)
    """
    runs = []
    for root, dirs, files in os.walk(method_dir):
        # Look for .hydra directory in current folder
        if ".hydra" in dirs:
            config_path = Path(root) / ".hydra" / "config.yaml"
            # Look for checkpoint file in the chkpts subdirectory.
            chkpt_dir = Path(root) / "chkpts"
            if chkpt_dir.exists():
                # You can customize the selection logic if multiple checkpoints exist.
                # Here, we choose the checkpoint file with "last.ckpt" in its name.
                ckpt_files = list(chkpt_dir.glob("*last.ckpt"))
                if ckpt_files:
                    ckpt_path = str(ckpt_files[0].resolve())
                    # Use the directory name as part of the model name to reflect hyperparameters.
                    if "adversarial" in root:
                        model_name = "adversarial"
                    elif "cmmd" in root:
                        model_name = "cmmd"
                    else:
                        model_name = "baseline"
                    runs.append((model_name, str(config_path.resolve()), ckpt_path))
                else:
                    LOG.warning(f"No checkpoint found in {chkpt_dir}")
            else:
                LOG.warning(f"No chkpts folder in {root}")
    return runs


def main():
    # Create output directory for CSV files
    output_dir = Path("evaluation/confounding_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory (optional)
    # os.chdir(output_dir)
    LOG.info(f"Results will be saved in {output_dir.resolve()}")

    # Define methods and device
    method_names = ["adversarial", "cmmd"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    for method in method_names:
        method_dir = Path(BASE_EXPERIMENT_DIR) / method
        if not method_dir.exists():
            LOG.warning(f"Method directory {method_dir} does not exist, skipping.")
            continue

        runs = discover_runs(method_dir)
        LOG.info(f"Found {len(runs)} runs for method {method}.")
        results_list = []

        # Define the CSV path for this method.
        csv_path = output_dir / f"{method}_analysis_results.csv"
        # If the CSV already exists, load it to resume progress.
        if csv_path.exists():
            # try:
            #     df_existing = pd.read_csv(csv_path)
            #     results_list = df_existing.to_dict('records')
            #     LOG.info(f"Loaded {len(results_list)} existing results from {csv_path}.")
            # except Exception as e:
            #     LOG.error(f"Failed to load existing CSV {csv_path}: {e}")

            # skip
            continue

        for idx, run in enumerate(runs):
            try:
                results, model_cfg = run_test(run, device=device, batch_size=batch_size)
                # Extract hyperparameters from the config
                hparams = extract_hparams(model_cfg, method)
                # Combine the run name, hyperparameters, and metrics in one flat dictionary.
                combined_result = {"model_name": results["model_name"]}
                combined_result.update(hparams)
                combined_result["bacc"] = results["bacc"]
                combined_result["roc_auc"] = results["roc_auc"]
                combined_result["ctf_mean_var_pred"] = results["ctf_mean_var_pred"]
                combined_result["ctf_mean_var_hat"] = results["ctf_mean_var_hat"]

                results_list.append(combined_result)
                LOG.info(f"Processed run {idx+1}/{len(runs)}: {results['model_name']}")
            except Exception as e:
                LOG.error(f"Error processing run {run[0]}: {e}")
                continue

            # Save intermediate results after each run.
            try:
                df_results = pd.DataFrame(results_list)
                print(df_results.head())
                df_results.to_csv(csv_path, index=False)
                LOG.info(f"Saved intermediate CSV for {method} to {csv_path}")
            except Exception as e:
                LOG.error(f"Error saving intermediate CSV for {method}: {e}")

        # Final message per method.
        if results_list:
            LOG.info(f"Completed processing {len(results_list)} runs for {method}. Final CSV saved at {csv_path}")
        else:
            LOG.info(f"No results collected for {method}.")
    LOG.info("All methods processed.")

if __name__ == "__main__":
    main()
