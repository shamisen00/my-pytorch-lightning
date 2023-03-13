# %%
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

# %%
cli = LightningCLI(DemoModel, BoringDataModule)