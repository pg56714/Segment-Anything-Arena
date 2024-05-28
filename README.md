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
create weights folder

cd weights

download https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt

put the downloaded file in the weights folder

### Running the Project

```
python app.py
```
