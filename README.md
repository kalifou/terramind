[![Website](https://img.shields.io/badge/Website-TerraMind-0F62FE)](https://ibm.github.io/terramind/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.11171-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.11171)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-IBM--ESA--Geospaital-FFD21E?logo=huggingface)](https://huggingface.co/ibm-esa-geospatial)
[![Downloads](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/IBM/terramind/webpage/assets/hf_monthly_downloads.json)](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
[![Code](https://img.shields.io/badge/Model_code-TerraTorch-EE4B2B?logo=github)](https://github.com/IBM/terratorch/tree/main/terratorch/models/backbones/terramind)
[![IBMblog](https://img.shields.io/badge/Blog-IBM-0F62FE)](https://research.ibm.com/blog/terramind-esa-earth-observation-model)
[![ESAblog](https://img.shields.io/badge/Blog-ESA-113145)](https://www.esa.int/Applications/Observing_the_Earth/ESA_and_IBM_collaborate_on_TerraMind)
[![Challenge](https://img.shields.io/badge/Website-Blue--sky_Challenge-0F62FE)](https://huggingface.co/spaces/ibm-esa-geospatial/challenge)

[//]: # (Weekly updates of downloads. See .github/workflows/hf-downloads.yml for configuration.)

# TerraMind 1.0

TerraMind is the first any-to-any generative foundation model for Earth Observation, build by IBM, ESA Φ-lab, and the FAST-EO project.
We pre-trained a [base version](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) and a [large version](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large) of TerraMind, both open-sourced on HuggingFace. 
The models are fully integrated into the fine-tuning toolkit [TerraTorch](https://ibm.github.io/terratorch/).

This repo presents code examples for fine-tuning TerraMind, using the Thinking-in-Modalities approach, and for any-to-any generations.
We refer to [Hugging Face](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) and [arXiv](https://arxiv.org/abs/2504.11171) for more detailed information. 

![terramind_architecture.png](assets%2Fterramind_architecture.png)

## Setup

Download or clone this repo and create a new environment with the latest version of TerraTorch.
```shell
python -m venv venv # use python 3.10 or higher
source venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/IBM/terratorch.git
pip install jupyter gdown tensorboard # required for notebook examples
pip install diffusers==0.30.0  # required for TerraMind generations
```

Note: We fixed an error in the multimodal dataset. 
Please install `terratorch` from main via `pip install git+https://github.com/IBM/terratorch.git` until a new version is released. 

## Fine-tuning

You can fine-tune TerraMind without any code using a Lightning config and [TerraTorch](https://ibm.github.io/terratorch/): 

```shell
terratorch fit -c <terramind_config.yaml>
```

For testing the fine-tuned TerraMind model, run:
```shell
terratorch test -c <terramind_config.yaml> --ckpt_path <path/to/your/checkpoint.ckpt>
```

We provide two config examples for [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) and [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars):

- [terramind_v1_base_sen1floods11.yaml](configs%2Fterramind_v1_base_sen1floods11.yaml)

- [terramind_v1_base_burnscars.yaml](configs%2Fterramind_v1_base_burnscars.yaml)

We use the `GenericMultiModalDataModule` in the Sen1Floods11 example and the standard `GenericNonGeoSegmentationDataModule` for the single-modal Burn Scars dataset.
We simplified the dataset folder structure compared to the original datasets. You can either adjust the paths in the config for the original datasets or download the updated version with the code in the notebooks.
The relevant parts of the config are explained in more detail in this notebook example: 

- [terramind_v1_base_sen1floods11.ipynb](notebooks%2Fterramind_v1_base_sen1floods11.ipynb)
  ([Open in Colab](https://colab.research.google.com/github/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb))


## Thinking-in-Modalities

TerraMind introduces a new Thinking-in-Modalities (TiM) approach, where other modalities are predicted as an intermediate steps.
Then, the fine-tuned encoder uses both raw inputs and the generated modalities. 
You simply need to add the suffix `_tim` to the model name and optionally define the TiM modalities:

```yaml
      backbone: terramind_v1_base_tim
      backbone_tim_modalities:
        - LULC  # default TiM modality
```

We share an example config for TiM fine-tuning here: [terramind_v1_base_tim_lulc_sen1floods11.yaml](configs%2Fterramind_v1_base_tim_lulc_sen1floods11.yaml). 
We refer to our paper for a more detailed explanation of the TiM approach.

## Any-to-any generation

TerraMind can perform any-to-any generation based on varying combinations of inputs.
You can test the generation capabilities with this notebook: [terramind_any_to_any_generation.ipynb](notebooks%2Fterramind_any_to_any_generation.ipynb).

If you are only interested in generating a single modality from another one, [terramind_generation.ipynb](notebooks%2Fterramind_generation.ipynb) provides a simplified version of the generation code.

We provide some examples images from the TerraMesh validation split in [examples/](examples).

## Tokenizer

TerraMind uses six tokenizer for pre-training and generation. 
We provide some example code for using the tokenizer in [terramind_tokenizer_reconstruction.ipynb](notebooks%2Fterramind_tokenizer_reconstruction.ipynb).

## Challenge

Already working with TerraMind? Submit your use case to the [TerraMind Blue-Sky Challenge](https://huggingface.co/spaces/ibm-esa-geospatial/challenge), a bi-monthly award spotlighting the boldest, most imaginative ways using TerraMind.

## Citation

If you use TerraMind in your research, please cite the [TerraMind](https://arxiv.org/abs/2504.11171) pre-print.

```text
@article{jakubik2025terramind,
  title={TerraMind: Large-Scale Generative Multimodality for Earth Observation},
  author={Jakubik, Johannes and Yang, Felix and Blumenstiel, Benedikt and Scheurer, Erik and Sedona, Rocco and Maurogiovanni, Stefano and Bosmans, Jente and Dionelis, Nikolaos and Marsocci, Valerio and Kopp, Niklas and others},
  journal={arXiv preprint arXiv:2504.11171},
  year={2025}
}
```