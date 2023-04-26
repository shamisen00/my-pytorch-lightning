import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List
import random

from skimage.color import lab2rgb, rgb2lab
from PIL import Image
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import torch
import torchvision.transforms as transforms
import torchvision
# import webdataset as wds

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)

@rank_zero_only
def create_wds(cfg: DictConfig) -> None:
    """Creates wds dataset in `cfg.data.datasets`."""

    dataset_root = Path(cfg.paths.data_dir)
    shard_path = dataset_root / 'shards'

    if not shard_path.exists():
        shard_size = cfg.data.shard_size * 1000**2

        file_paths = [path for path in (dataset_root / 'train').glob('*')]

        random.shuffle(file_paths)

        shard_path.mkdir(exist_ok=True)
        shard_filename = str(shard_path / 'shards-%05d.tar')

        with wds.ShardWriter(shard_filename, maxsize=shard_size) as sink, tqdm(file_paths) as pbar:

            for file_path in pbar:

                key_str = file_path.stem

                gt_file_path = str(dataset_root / 'gt' / f'{key_str}.jpg')

                with open(file_path, 'rb') as raw_bytes:
                    train_buffer = raw_bytes.read()

                with open(gt_file_path, 'rb') as raw_bytes:
                    gt_buffer = raw_bytes.read()

                sink.write({
                    "__key__": key_str,
                    "train.jpg": train_buffer,
                    "gt.jpg": gt_buffer,
                })
    else:
        pass


def rgb2lab_torch(rgb_tensor: torch.Tensor) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    rgb_tensor = torch.permute(rgb_tensor, (1, 2, 0))
    lab = rgb2lab(rgb_tensor)
    lab_tensor = to_tensor(lab)

    return lab_tensor


def lab2rgb_torch(lab_tensor: torch.Tensor) -> Image:
    to_tensor = transforms.ToTensor()
    # to_pil = transforms.ToPILImage()

    # print("\n input of output a", "\n", lab_tensor[1, :, :].min())
    # print("\n input of output a", "\n", lab_tensor[1:, :, :].max())

    # print("\n input of output b", "\n", lab_tensor[2, :, :].min())
    # print("\n input of output b", "\n", lab_tensor[2:, :, :].max())

    # print("\n input of output renomal", "\n", lab_tensor[2, :, :].min(), "\n")
    # print("\n input of output renomal", "\n", lab_tensor[2:, :, :].max(), "\n")
    lab_tensor = torch.permute(lab_tensor, (1, 2, 0))
    lab = lab2rgb(lab_tensor)
    rgb_tensor = to_tensor(lab)
    # print("\n output","\n",rgb_tensor[2, :, :].min())
    # print("\n output","\n",rgb_tensor[2, :, :].max())

    return rgb_tensor


def normalize(im):
    im[0, :, :] = im[0, :, :] / 100

    min_input, max_input = -90, 100
    min_output, max_output = 0, 1

    im[1:, :, :] = (im[1:, :, :] - min_input) * (max_output - min_output) / (max_input - min_input) + min_output
    return im


def de_normalize(im):
    im[0, :, :] = im[0, :, :] * 100

    min_input, max_input = 0, 1
    min_output, max_output = -90, 100

    im[1:, :, :] = (im[1:, :, :] - min_input) * (max_output - min_output) / (max_input - min_input) + min_output
    return im


def _to_grid(images):
    return torchvision.utils.make_grid(tensor=images)


def to_rgb(image):
    image = de_normalize(image)
    image = lab2rgb_torch(image.cpu())
    image = _to_grid(image)

    return image
