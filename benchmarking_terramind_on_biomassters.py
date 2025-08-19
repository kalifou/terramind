#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:42:30 2025

@author: trao_ka
"""
import ipdb

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

TIM_MODALITIES_LIST = ['sen2', 'sen1', 'caption', 'sen1rtc', 
                       'dem', 'lulc', 'ndvi', 'caption', 'coords'
                       ]

def list_of_str(arg):
    result = list()
    if len(arg) > 0:
        result = list(map(str, arg.split(',')))
    return result

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

@click.option("-tst", "--test_mode", 
              is_flag=True,
              help='Activate test mode.')

@click.option("-ltim", "--list_of_tim_modalities", 
              type=list_of_str, 
              default="",
              help='Activate Thinking in Modality (TiM).')

@click.option("-pthd", '--data_path', 
              type=str, 
              default="data/subsample_biomasters_500/", 
              help='Path to Biomassters dataset')

@click.option("-pthl", '--logs_path', 
              type=str, 
              default="results/terramind_on_biomassters/", 
              help='Path for saving the results and logs of the experiments.')

@click.option("-sd", '--seed', 
              type=int, 
              required=False, 
              default=0,
              help='Experimental seed.')




def main(max_epochs, 
         batch_size, 
         sensor, 
         backbone_size, 
         freeze_the_backbone, 
         test_mode, 
         data_path,
         logs_path,
         list_of_tim_modalities,
         seed):

    print("\n\nExperimental protocol:")
    print("sensor: {}".format(sensor))
    print("max_epochs: {}".format(int(max_epochs)))
    print("batch_size: {}".format(int(batch_size)))
    print("test_mode: {}".format(int(test_mode)))    
    print("Thinking in Modality (TiM): {}".format(list_of_tim_modalities))
    print("freeze_the_backbone: {}".format(int(freeze_the_backbone)))
    print("backbone_size: {}".format(backbone_size))
    print("seed: {}".format(int(seed)))    
    print("data_path: {}".format(data_path))    
    print("logs_path: {}\n\n".format(logs_path))
    
    assert backbone_size in ["base", "large"]
    assert sensor in ["s2", "s1"]
    assert isinstance(test_mode, bool)
    assert isinstance(freeze_the_backbone, bool)
    assert isinstance(list_of_tim_modalities, list)
    for tim_i in list_of_tim_modalities:
        assert tim_i in TIM_MODALITIES_LIST
    
    num_of_channels = 12
    image_size = 256
    
    
    if sensor == "s2":
        backbone_modalities = ["S2L2A"]
    elif sensor == "s1":
        num_of_channels = 4
        backbone_modalities = ["S1GRD"]
    elif sensor == "fusion":
        num_of_channels + 4
        backbone_modalities = ["S2L2A", "S1GRD"]
    else:
        pass
    
    if backbone_size == "base":
        neck_indices = [2, 5, 8, 11] # indices for terramind_v1_base
    elif backbone_size == "large":
        neck_indices = [5, 11, 17, 23] # indices for terramind_v1_large
    
    
    terramind_backone = "terramind_v1_" + backbone_size
    
    model_args = {"backbone":terramind_backone,
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
                  }
    
    if list_of_tim_modalities != list():
        terramind_backone += "_tim"
        
        model_args["backbone"] = terramind_backone
        model_args["backbone_tim_modalities"] = list_of_tim_modalities
    
    #ipdb.set_trace(context=25)
    
    regressor_model = PixelwiseRegressionTask(freeze_backbone=freeze_the_backbone,
                                              model_args=model_args,
                                              model_factory="EncoderDecoderFactory",
                                              plot_on_val=False
                                              )
    # Fix the input features dimentions in the tim modality sampler to account for Biomassters input data dimensions    
    target_in_features = image_size * num_of_channels

    sensor_key_for_embeddings = "untok_sen2l2a@224"
    
    if backbone_modalities == ["S2L2A"]:
        sensor_key_for_embeddings = "untok_sen2l2a@224"

    elif backbone_modalities == ["S2L1C"]:
        sensor_key_for_embeddings = "untok_sen2l1c@224"

    elif backbone_modalities == ["S1GRD"]:
        sensor_key_for_embeddings = "untok_sen1grd@224"

    else:
        pass
    
    #ipdb.set_trace(context=150)
    
    target_out_features_size = regressor_model.model.encoder.\
        encoder_embeddings[sensor_key_for_embeddings].proj.out_features
    
    regressor_model.model.encoder.\
        encoder_embeddings[sensor_key_for_embeddings].proj = \
            torch.nn.Linear(in_features=target_in_features, 
                            out_features=target_out_features_size,
                            bias=False)
            
    # Fix the input features dimentions in the tim modality sampler to account for Biomassters input data dimensions
    if list_of_tim_modalities != list():
        target_in_features = image_size * num_of_channels
        target_out_features_size = regressor_model.model.encoder.sampler.model.\
            encoder_embeddings[sensor_key_for_embeddings].proj.out_features
        
        regressor_model.model.encoder.sampler.model.\
            encoder_embeddings[sensor_key_for_embeddings].proj = \
                torch.nn.Linear(in_features=target_in_features, 
                                out_features=target_out_features_size,
                                bias=False)  
    
    #ipdb.set_trace(context=25)
    
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
    
    pl.seed_everything(seed)
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    path_to_logs_local = logs_path + "/" + terramind_backone + "/freeze_backbone_" + str(freeze_the_backbone) \
        + "/"+ sensor + "/" + str(max_epochs) + "/" + dt_string
    if list_of_tim_modalities == list():
        path_to_logs_local = logs_path + "/" + terramind_backone + "/freeze_backbone_" + str(freeze_the_backbone) \
            + "/"+ sensor + "/" + str(max_epochs) + "/" + dt_string
    else:
        path_to_logs_local = logs_path + "/" + terramind_backone + "/freeze_backbone_" + str(freeze_the_backbone) \
            + "/"+ sensor + "/ltim"

        temporary_str = ""
        path_to_logs_local += temporary_str.join("_" + tim_i for tim_i in list_of_tim_modalities)
        path_to_logs_local += "/"+ str(max_epochs) + "/" + dt_string
        
    path_to_save_logs = path_to_logs_local
    
    
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
