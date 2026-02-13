"""
GridSearch Hyperparameter Optimization
=======================================
Systematic hyperparameter search for diabetic retinopathy model.
Explores combinations of learning rates, dropout, architectures,
batch sizes, and loss functions to find the optimal configuration.

Results are logged to MLflow and saved as a CSV report.
"""

import copy
import itertools
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml
import torch

from src.training.train import Trainer, load_config

logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT SEARCH SPACE
# ============================================================================
DEFAULT_SEARCH_SPACE = {
    "model.architecture": ["efficientnet_b0", "efficientnet_b3"],
    "model.dropout": [0.3, 0.4, 0.5],
    "training.learning_rate": [1e-4, 3e-4, 1e-3],
    "training.optimizer": ["adamw"],
    "training.batch_size": [16, 32],
}


def _set_nested_key(d: Dict, key: str, value: Any) -> Dict:
    """
    Set a nested dictionary value using dot notation.

    Example:
        _set_nested_key(config, "model.dropout", 0.5)
        → config["model"]["dropout"] = 0.5
    """
    keys = key.split(".")
    current = d
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = value
    return d


def _get_nested_key(d: Dict, key: str) -> Any:
    """Get a nested dictionary value using dot notation."""
    keys = key.split(".")
    current = d
    for k in keys:
        current = current[k]
    return current


class GridSearchOptimizer:
    """
    GridSearch hyperparameter optimizer.

    Systematically trains models with every combination of hyperparameters
    in the search space, evaluates each on the validation set, and reports
    the best configuration.

    Args:
        base_config_path: Path to the base config.yaml.
        search_space: Dictionary mapping dotted config keys to lists of values.
            Example: {"model.dropout": [0.3, 0.4, 0.5]}
        max_epochs_per_trial: Maximum epochs per grid search trial (shorter
            than full training for speed).
        output_dir: Directory to save search results.
    """

    def __init__(
        self,
        base_config_path: str = "config/config.yaml",
        search_space: Optional[Dict[str, List[Any]]] = None,
        max_epochs_per_trial: int = 15,
        output_dir: str = "outputs/grid_search",
    ):
        self.base_config = load_config(base_config_path)
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.max_epochs_per_trial = max_epochs_per_trial
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all combinations
        self.param_names = list(self.search_space.keys())
        self.param_values = list(self.search_space.values())
        self.combinations = list(itertools.product(*self.param_values))

        logger.info(f"GridSearch initialized with {len(self.combinations)} combinations")
        for name, values in self.search_space.items():
            logger.info(f"  {name}: {values}")

    def run(self) -> pd.DataFrame:
        """
        Execute the full grid search.

        Trains a model for each hyperparameter combination and records
        validation metrics.

        Returns:
            DataFrame with all trial results, sorted by the primary metric.
        """
        results = []
        total = len(self.combinations)

        logger.info("=" * 60)
        logger.info(f"GRID SEARCH: {total} trials")
        logger.info("=" * 60)

        for trial_idx, combo in enumerate(self.combinations):
            trial_params = dict(zip(self.param_names, combo))

            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial_idx + 1}/{total}")
            logger.info(f"Parameters: {trial_params}")
            logger.info(f"{'='*60}")

            try:
                trial_result = self._run_single_trial(trial_idx, trial_params)
                results.append(trial_result)
                logger.info(
                    f"Trial {trial_idx + 1} completed — "
                    f"Best Recall: {trial_result.get('best_val_recall', 0):.4f}, "
                    f"AUC: {trial_result.get('best_val_auc', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"Trial {trial_idx + 1} FAILED: {e}")
                trial_result = {**trial_params, "status": "failed", "error": str(e)}
                results.append(trial_result)

            # Free GPU memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Build results DataFrame
        results_df = pd.DataFrame(results)

        # Sort by best recall (primary metric)
        if "best_val_recall" in results_df.columns:
            results_df = results_df.sort_values(
                "best_val_recall", ascending=False
            ).reset_index(drop=True)

        # Save results
        results_path = self.output_dir / "grid_search_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")

        # Print top results
        self._print_summary(results_df)

        # Save best config
        if len(results_df) > 0 and "status" in results_df.columns:
            successful = results_df[results_df["status"] == "completed"]
            if len(successful) > 0:
                best_row = successful.iloc[0]
                self._save_best_config(best_row)

        return results_df

    def _run_single_trial(
        self, trial_idx: int, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single grid search trial.

        Args:
            trial_idx: Trial index (for naming).
            trial_params: Dictionary of hyperparameters for this trial.

        Returns:
            Dictionary with trial parameters and results.
        """
        # Deep-copy base config and apply trial params
        config = copy.deepcopy(self.base_config)
        for key, value in trial_params.items():
            _set_nested_key(config, key, value)

        # Override max epochs for faster search
        config["training"]["max_epochs"] = self.max_epochs_per_trial
        config["training"]["min_epochs"] = min(5, self.max_epochs_per_trial)
        config["training"]["early_stopping"] = True
        config["training"]["patience"] = 5

        # Unique checkpoint dir per trial
        trial_dir = self.output_dir / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config["logging"]["checkpoint_dir"] = str(trial_dir)

        # Disable MLflow for individual trials (too noisy)
        config["logging"]["use_mlflow"] = False

        start_time = time.time()

        # Train
        trainer = Trainer(config)
        train_results = trainer.fit()

        elapsed = time.time() - start_time

        # Collect results
        history = train_results["history"]
        result = {
            **trial_params,
            "status": "completed",
            "best_val_recall": max(history.get("val_recall", [0])),
            "best_val_auc": max(history.get("val_auc", [0])),
            "best_val_f1": max(history.get("val_f1", [0])),
            "best_val_accuracy": max(history.get("val_accuracy", [0])),
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("inf"),
            "epochs_trained": len(history.get("train_loss", [])),
            "training_time_min": elapsed / 60,
        }

        # Clean up model from memory
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    def _print_summary(self, results_df: pd.DataFrame):
        """Print a formatted summary of grid search results."""
        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH RESULTS SUMMARY")
        logger.info("=" * 80)

        successful = results_df[results_df.get("status", "completed") == "completed"]
        if len(successful) == 0:
            logger.warning("No successful trials!")
            return

        # Top 5 by recall
        display_cols = self.param_names + [
            "best_val_recall",
            "best_val_auc",
            "best_val_f1",
            "epochs_trained",
            "training_time_min",
        ]
        existing_cols = [c for c in display_cols if c in successful.columns]
        top5 = successful.head(5)[existing_cols]

        logger.info("\nTop 5 configurations (by validation recall):")
        logger.info(top5.to_string(index=False))

        # Best config
        best = successful.iloc[0]
        logger.info(f"\n★ BEST CONFIGURATION:")
        for param in self.param_names:
            if param in best:
                logger.info(f"  {param}: {best[param]}")
        logger.info(f"  Recall: {best.get('best_val_recall', 'N/A'):.4f}")
        logger.info(f"  AUC:    {best.get('best_val_auc', 'N/A'):.4f}")
        logger.info(f"  F1:     {best.get('best_val_f1', 'N/A'):.4f}")

    def _save_best_config(self, best_row: pd.Series):
        """Save the best configuration as a YAML file."""
        best_config = copy.deepcopy(self.base_config)
        for param in self.param_names:
            if param in best_row:
                _set_nested_key(best_config, param, best_row[param])

        # Restore full training settings
        best_config["training"]["max_epochs"] = self.base_config["training"]["max_epochs"]

        best_config_path = self.output_dir / "best_config.yaml"
        with open(best_config_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)

        logger.info(f"\nBest config saved to {best_config_path}")
        logger.info("Run full training with: python -m src.training.train --config "
                     f"{best_config_path}")


def main():
    """CLI entry point for grid search."""
    import argparse

    parser = argparse.ArgumentParser(description="GridSearch Hyperparameter Optimization")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=15,
        help="Max epochs per trial (default: 15)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/grid_search",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    optimizer = GridSearchOptimizer(
        base_config_path=args.config,
        max_epochs_per_trial=args.max_epochs,
        output_dir=args.output_dir,
    )
    results = optimizer.run()
    print(f"\nGrid search complete. {len(results)} trials evaluated.")


if __name__ == "__main__":
    main()
