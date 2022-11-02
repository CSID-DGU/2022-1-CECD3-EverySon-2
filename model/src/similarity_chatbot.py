# only for similarity based chatbot
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import DataLoader
from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Predicts given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    log.info(f"Instantiating dataloaders")
    dataloaders: DataLoader = hydra.utils.instantiate(cfg.datamodule).predict_dataloader()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "dataloaders": dataloaders,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting predicting!")
    preds = sum(trainer.predict(model=model,
                            dataloaders=dataloaders,
                            ckpt_path=cfg.ckpt_path), [])
    log.info("Saving Result!")
    save_result(
        data_dir=cfg.datamodule.pred_data_dir,
        output_dir=cfg.paths.output_dir,
        predictions=preds,
        )

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


def save_result(data_dir, output_dir, predictions):
    import pandas as pd
    import os.path

    df = pd.read_csv(data_dir, encoding="UTF-8")
    df["prediction"] = predictions
    df.to_csv(os.path.join(output_dir, "output.csv"), index=False)
    log.info(f"input  : {(df['text'] + ' [SEP] ' + df['label']).tolist()}")
    log.info(f"output : {df['prediction'].tolist()}")


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="chatbot_pred.yaml")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
