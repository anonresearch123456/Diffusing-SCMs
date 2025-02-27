import os
import sys
import logging
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Import simulation-specific modules (adjust imports as needed)
from src.data.simulation_data import SimDataModule
from src.simulation.debiasing import AdversarialPredictor, cmmdRegularizedPredictor, MetaDataPrediction

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Base experiment directory for simulation experiments
BASE_SIM_EXPERIMENT_DIR = ""

# Define the simulation dataset (here, only one synthetic dataset is assumed)
DATASET = [("standard_simulation", "")]


def extract_hparams(config, method):
    """
    Extract key hyperparameters from the config depending on the debiasing method.
    For adversarial, we extract lambda_feature, lambda_protected, etc.
    For cmmd, we extract cmmd_lambda, inverted_bandwidth, etc.
    """
    hparams = {}
    pred_cfg = config.get("predictor", {})
    if method == "adversarial":
        hparams["lambda_feature"] = pred_cfg.get("lambda_feature")
        hparams["lambda_protected"] = pred_cfg.get("lambda_protected")
        hparams["protected_pred_head"] = pred_cfg.get("protected_pred_head")
    elif method == "cmmd":
        hparams["cmmd_lambda"] = pred_cfg.get("cmmd_lambda")
        hparams["inverted_bandwidth"] = pred_cfg.get("inverted_bandwidth")
    else:
        hparams["note"] = "baseline/default hyperparams"
    return hparams


def run_test(model_triplet: tuple, device: str = "cuda:0", batch_size: int = 128):
    """
    Run the test pipeline for one simulation experiment.
    Loads the model (from its config and checkpoint) and runs testing on the simulation dataset.
    Returns a dict of metrics and the model configuration.
    """
    model_name, config_path, ckpt_path = model_triplet
    LOG.info(f"Testing model: {model_name}")

    model_cfg = OmegaConf.load(config_path)

    # Load model based on the debiasing method indicated in the configuration.
    if "adversarial" in model_cfg.debiasing_method:
        model = AdversarialPredictor.load_from_checkpoint(ckpt_path)
    elif "cmmd" in model_cfg.debiasing_method:
        model = cmmdRegularizedPredictor.load_from_checkpoint(ckpt_path)
    else:
        model = MetaDataPrediction.load_from_checkpoint(ckpt_path)

    model = model.to(device)
    model.eval()
    model.freeze()

    # Load the simulation dataset
    dataset_name, dataset_path = DATASET[0]
    data_module = SimDataModule(dataset_path, batch_size=batch_size, num_workers=4)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # Run inference and collect metrics
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.test_step(batch, batch_idx)
    metrics = model.on_test_epoch_end()
    LOG.info(f"Metrics for model {model_name}: {metrics}")

    return metrics, model_cfg


def discover_runs(method_dir: Path):
    """
    Discover all experiment runs under a given method directory.
    Each run should have a .hydra/config.yaml and a chkpts directory with a checkpoint.
    Returns a list of tuples: (model_name, config_path, checkpoint_path)
    """
    runs = []
    for root, dirs, files in os.walk(method_dir):
        if ".hydra" in dirs:
            config_path = Path(root) / ".hydra" / "config.yaml"
            chkpt_dir = Path(root) / "chkpts"
            if chkpt_dir.exists():
                ckpt_files = list(chkpt_dir.glob("*last.ckpt"))
                if ckpt_files:
                    ckpt_path = str(ckpt_files[0].resolve())
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
    output_dir = Path("evaluation/simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Results will be saved in {output_dir.resolve()}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    # Define the debiasing methods to explore
    method_names = ["adversarial", "cmmd"]

    for method in method_names:
        method_dir = Path(BASE_SIM_EXPERIMENT_DIR) / method
        if not method_dir.exists():
            LOG.warning(f"Method directory {method_dir} does not exist, skipping.")
            continue

        runs = discover_runs(method_dir)
        LOG.info(f"Found {len(runs)} runs for method {method}.")
        results_list = []
        csv_path = output_dir / f"{method}_analysis_results.csv"

        # If the CSV already exists, skip this method.
        if csv_path.exists():
            LOG.info(f"CSV file {csv_path} already exists; skipping method {method}.")
            continue

        for idx, run in enumerate(runs):
            try:
                metrics, model_cfg = run_test(run, device=device, batch_size=batch_size)
                hparams = extract_hparams(model_cfg, method)
                # Combine run info, hyperparameters, and metrics into a flat dictionary.
                combined_result = {"model_name": run[0]}
                combined_result.update(hparams)
                combined_result.update(metrics)
                results_list.append(combined_result)
                LOG.info(f"Processed run {idx+1}/{len(runs)}: {run[0]}")
            except Exception as e:
                LOG.error(f"Error processing run {run[0]}: {e}")
                continue

            # Save intermediate CSV results after each run.
            try:
                df_results = pd.DataFrame(results_list)
                df_results.to_csv(csv_path, index=False)
                LOG.info(f"Saved intermediate CSV for {method} to {csv_path}")
            except Exception as e:
                LOG.error(f"Error saving intermediate CSV for {method}: {e}")

        if results_list:
            LOG.info(f"Completed processing {len(results_list)} runs for {method}. Final CSV saved at {csv_path}")
        else:
            LOG.info(f"No results collected for {method}.")
    LOG.info("All methods processed.")


if __name__ == "__main__":
    main()
