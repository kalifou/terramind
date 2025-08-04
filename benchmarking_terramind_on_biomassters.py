#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:42:30 2025

@author: trao_ka
"""

import click
from datetime import datetime

# from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models import EncoderDecoderFactory
from grunblick import BiomasstersDataModule
from grunblick.utils import get_dataloaders

import torch
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY

import lightning.pytorch as pl
from terratorch.tasks import PixelwiseRegressionTask

from lightning.pytorch.callbacks import RichProgressBar

@click.command()

@click.option("-me", '--max_epochs', 
              type=int, 
              required=False, 
              default=2,
              help='Number of epochs.')

@click.option("-bs", '--batch_size', 
              type=int, 
              required=False, 
              default=8,
              help='Batch size.')

@click.option("-sr",'--sensor', 
              type=str, 
              default="s2", 
              help='Sensor to use, among: {s1, s2}.')

@click.option("-bcsz",'--backbone_size', 
              type=str, 
              default="base", 
              help='Size of the terramind backbone, among: {base, large}.')

@click.option("-fzb", '--freeze_the_backbone', 
              is_flag=True,
              help='Activate the freezing of the backbone while finetuning.')

@click.option("-tm", "--test_mode", 
              is_flag=True,
              help='Activate test mode.')

@click.option("-pthd", '--data_path', 
              type=str, 
              default="data/subsample_biomasters_500/", 
              help='Path to Biomassters dataset')

@click.option("-pthl", '--logs_path', 
              type=str, 
              default="results/terramind_on_biomassters/", 
              help='Path for saving the results and logs of the experiments.')

def main(max_epochs, 
         batch_size, 
         sensor, 
         backbone_size, 
         freeze_the_backbone, 
         test_mode, 
         data_path,
         logs_path):

    #import ipdb
    #ipdb.set_trace(context=35)
    print("\n\nExperimental protocol:")
    print("sensor: {}".format(sensor))
    print("max_epochs: {}".format(int(max_epochs)))
    print("batch_size: {}".format(int(batch_size)))
    print("test_mode: {}".format(int(test_mode)))
    print("freeze_the_backbone: {}".format(int(freeze_the_backbone)))
    print("backbone_size: {}".format(backbone_size))
    print("data_path: {}".format(data_path))    
    print("logs_path: {}\n\n".format(logs_path))
    
    assert backbone_size in ["base", "large"]
    assert sensor in ["s2", "s1"]
    
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
        
    if backbone_size == "base":
        neck_indices = [2, 5, 8, 11] # indices for terramind_v1_base
    elif backbone_size == "large":
        neck_indices = [5, 11, 17, 23] # indices for terramind_v1_large
        
    terramind_backone = "terramind_v1_" + backbone_size
    
    regressor_model = PixelwiseRegressionTask(freeze_backbone=freeze_the_backbone,
                                               model_args={"backbone":terramind_backone,
                                                           "backbone_pretrained": True,
                                                           "backbone_modalities": backbone_modalities,
                                                          
                                                             # Necks 
                                                             "necks": [
                                                                 {
                                                                     "name": "SelectIndices",
                                                                     "indices": neck_indices
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
    
    
    root = data_path
    
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
                                              output_format="terratorch")
    
    pl.seed_everything(0)
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    path_to_save_logs = (
        logs_path
        + "/"
        + terramind_backone
        + "/"
        "freeze_backbone_" + str(freeze_the_backbone)
        + "/"
        + sensor
        + "/"
        + str(max_epochs)
        + "/"
        + dt_string
    )
    
    
    # # By default, TerraTorch saves the model with the best validation loss. You can overwrite this by defining a custom ModelCheckpoint, e.g., saving the model with the highest validation mIoU.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=path_to_save_logs + "/checkpoints/",
    #     mode="max",
    #     monitor="val/Multiclass_Jaccard_Index",  # Variable to monitor
    #     filename="best-mIoU",
        save_weights_only=True,
    )
    
    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        logger=True,  # Uses TensorBoard by default
        max_epochs=max_epochs,  # For demos
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, RichProgressBar(leave=True)],
        default_root_dir=path_to_save_logs,
    )
    
    trainer.fit(regressor_model, datamodule=local_datamodule)

    trainer.test(regressor_model,
                 datamodule=local_datamodule
                 )

    print("Finetuning and evaluation completed!")

if __name__ == "__main__":
    main()
