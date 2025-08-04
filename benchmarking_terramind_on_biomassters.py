#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:42:30 2025

@author: trao_ka
"""

# from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models import EncoderDecoderFactory
from grunblick import BiomasstersDataModule
from grunblick.utils import get_dataloaders

import torch
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY

import lightning.pytorch as pl
from terratorch.tasks import PixelwiseRegressionTask

from lightning.pytorch.callbacks import RichProgressBar


output_format = "terratorch"
batch_size = 16
sensor = "s2"
test_mode = True
max_epochs = 1

num_of_channels = 10
image_size = 256


if sensor == "s2":
    backbone_modalities = ["S2L1C"]
elif sensor == "s1":
    num_of_channels = 4
    backbone_modalities = ["S1GRD"]
else:
    num_of_channels + 4
    backbone_modalities = ["S2L1C", "S1GRD"]
    
regressor_model = PixelwiseRegressionTask(freeze_backbone=True,
                                           model_args={"backbone": "terramind_v1_large",
                                                       "backbone_pretrained": True,
                                                       "backbone_modalities": ["S2L1C"], #, "S1GRD"],
                                                      
                                                         # Necks 
                                                         "necks": [
                                                             {
                                                                 "name": "SelectIndices",
                                                                 "indices": [2, 5, 8, 11] # indices for terramind_v1_base
                                                                 # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
                                                             },
                                                             {"name": "ReshapeTokensToImage",
                                                              "remove_cls_token": False},  # TerraMind is trained without CLS token, which neads to be specified.
                                                             {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
                                                         ],
                                                         "decoder": "UNetDecoder",
                                                         "decoder_channels": [512, 256, 128, 64],
                                                         },
                                          model_factory="EncoderDecoderFactory",
                                          plot_on_val=False
                                          )
#import ipdb
#ipdb.set_trace(context=25)

target_in_features = image_size * num_of_channels
target_out_features_size = regressor_model.model.encoder.\
    encoder_embeddings["untok_sen2l1c@224"].proj.out_features

regressor_model.model.encoder.\
    encoder_embeddings["untok_sen2l1c@224"].proj = \
        torch.nn.Linear(in_features=target_in_features, 
                        out_features=target_out_features_size,
                        bias=False)



root = "/localhome/trao_ka/Documents/projects/gruenblick/repo_dlr_project/grunblick/data/subsample_biomasters_500/"

features_metadata = root + "/The_BioMassters_-_features_metadata.csv.csv"
# features_df = pd.read_csv(features_metadata)
train_img_dir = root + "/train_features/"
test_img_dir = root + "/test_features/"

train_label_dir = root + "/train_agbm/"
test_label_dir = root + "/test_agbm/"


local_datamodule = BiomasstersDataModule(features_metadata_definition=features_metadata,
                                          train_img_dir=train_img_dir,
                                          train_label_dir=train_label_dir,
                                          test_img_dir=test_img_dir,
                                          test_label_dir=test_label_dir,
                                          sensor_definition=sensor,
                                          test_mode_definition=test_mode,
                                          batch_size=batch_size,
                                          output_format=output_format)

pl.seed_everything(0)

# # By default, TerraTorch saves the model with the best validation loss. You can overwrite this by defining a custom ModelCheckpoint, e.g., saving the model with the highest validation mIoU.
# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     dirpath="output/terramind_base_biomassters/checkpoints/",
#     mode="max",
#     monitor="val/Multiclass_Jaccard_Index",  # Variable to monitor
#     filename="best-mIoU",
#     save_weights_only=True,
# )

# Lightning Trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    logger=True,  # Uses TensorBoard by default
    max_epochs=max_epochs,  # For demos
    #log_every_n_steps=1,
    callbacks=[RichProgressBar(leave=True)],
    default_root_dir="output/terramind_base_biomassters/",
)

trainer.fit(regressor_model, datamodule=local_datamodule)

# # # Train the model ⚡
# trainer.fit(
#      regressor_model,
#      train_dataloaders=local_train_dataloader,
#      val_dataloaders=local_valid_dataloader,
# )

trainer.test(regressor_model,
             #dataloaders=local_test_dataloader,
             datamodule=local_datamodule
             )


print("Hello")
