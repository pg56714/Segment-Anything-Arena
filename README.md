# EfficientViT-SAM vs EfficientSAM vs SAM

[HuggingFace Space](https://huggingface.co/spaces/pg56714/Segment-Anything-Arena)

## Getting Started

### Installation

Use Anaconda to create a new environment and install the required packages.

```
conda create -n samarena python=3.10 -y

conda activate samarena

pip install -r requirements.txt
```

## Models

| EfficientSAM-S                                                                                  | EfficientViT-SAM-XL1                                                            |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vits.pt.zip) | [Download](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt) |

create weights folder

cd weights

download and put the downloaded file in the weights folder

## Running the Project

```
python app.py
```
