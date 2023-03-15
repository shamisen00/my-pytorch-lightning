from pytorch_lightning.cli import LightningCLI


def main():
    # To enable variable interpolation, first install omegaconf:
    LightningCLI(save_config_filename="run.yaml",
                 save_config_multifile=True, save_config_overwrite=True,
                 parser_kwargs={"parser_mode": "omegaconf",
                                "fit": {"default_config_files": ["configs/config.yaml"]}})


if __name__ == "__main__":
    main()
