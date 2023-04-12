from lightning.pytorch.cli import LightningCLI
from src.models.picture_module import PictureModule


def main():
    # To enable variable interpolation, first install omegaconf:
    LightningCLI(
        model_class=PictureModule,
        parser_kwargs={
            "parser_mode": "omegaconf",
            "fit": {"default_config_files": ["configs/pictures.yaml"]}
            }
        )


if __name__ == "__main__":
    main()
