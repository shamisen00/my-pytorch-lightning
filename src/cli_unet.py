from lightning.pytorch.cli import LightningCLI
from src.models.unet_model import UNet


def main():
    # To enable variable interpolation, first install omegaconf:
    LightningCLI(
        save_config_kwargs={"overwrite": True},
        parser_kwargs={
            "parser_mode": "omegaconf",
            "fit": {"default_config_files": ["configs/unet.yaml"]},
            }
        )


if __name__ == "__main__":
    main()
