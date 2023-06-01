from typing import Optional, Tuple

from lightning.pytorch.callbacks import Callback
import plotly.graph_objects as go
from torchvision import transforms
import torch
import numpy as np

from src.utils.utils import lab2rgb_torch, de_normalize, get_mask


class SaveFigure(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            sample_losses = pl_module.sample_losses.compute().cpu()
            fig = go.Figure()
            print(len(sample_losses),"\n")

            fnames = np.array(pl_module.sample_fnames)
            suffixes = np.array(pl_module.sample_suffixes)
            print(len(fnames),"\nfnames")
            for suffix in pl_module.suffixes:
                mask = get_mask(suffix, pl_module.sample_suffixes)
                print(len(mask),"\nmask")
                fig.add_trace(
                    go.Scatter(
                        x=suffixes[mask],
                        y=sample_losses[mask],
                        text=[fname + "_" + suffix for fname, suffix in zip(fnames[mask], suffixes[mask])],
                        mode="markers",
                        hoverinfo="x+y+text",
                        name=f"{suffix}"
                    )
                    )
            fig.write_html(f"/workspace/{trainer.logger.run_id}.html")

            trainer.logger.experiment.log_figure(
                trainer.logger.run_id,
                figure=fig,
                artifact_file=f"{trainer.logger.run_id}.html"
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch == trainer.max_epochs - 1:
            y_hat = outputs["y_hat"]
            targets = outputs["targets"]
            fnames = batch[2][0]
            suffixes = batch[2][1]

            mae = torch.mean(torch.abs(targets - y_hat), dim=(1, 2, 3))

            print("\nfname_per",len(fnames),"\nfname_per")
            pl_module.sample_losses.update(mae)
            pl_module.sample_fnames.extend(fnames)
            pl_module.sample_suffixes.extend(suffixes)
