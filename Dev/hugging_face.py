# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:52:36 2024

@author: Richard
"""

from src.crackest.cracks import CrackPy, CrackPlot

# %s
cp = CrackPy(model=1)  # Model optimized also for pores

# %

imfile = r"Examples/Img/ID14_940_Image.png"  # Read a file
cp.get_mask(imfile)
# %%
cp.preview(mask="crack")
# %%

model_path = cp.model_path
print(model_path)

# %%
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class MyModel(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="https://huggingface.co/rievil/crackenpy",
    pipeline_tag="crack-segment",
    license="mit",
):
    def __init__(self, num_channels: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.param = nn.Parameter(torch.rand(num_channels, hidden_size))
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x + self.param)


# create model
config = {"num_channels": 3, "num_classes": 5}
model = MyModel(**config)

# save locally
model.save_pretrained("my-awesome-model")

# push to the hub
model.push_to_hub("your-hf-username/my-awesome-model")

# reload
model = MyModel.from_pretrained(model)

# %%
from huggingface_hub import (
    HfApi,
    HfFolder,
    Repository,
    create_repo,
    upload_file,
)
import os

# Your Hugging Face username and repo name
repo_id = "rievil/crackenpy"

# Path to your model file
model_path = "/Users/richarddvorak/Documents/Envs/base/lib/python3.12/site-packages/crackpy_models/resnext101_32x8d_N387_C5_310124.pt"  # Update this to your actual model path

# Your Hugging Face API token
api_token = (
    "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"  # Replace with your actual token
)

# Set up authentication with Hugging Face
HfFolder.save_token(api_token)

# Create or access the repository on Hugging Face
api = HfApi()
repo_url = api.create_repo(repo_id, exist_ok=True)

# Define the local directory to store the repository files
repo_dir = "./crackenpy_repo"

# Clone the Hugging Face repo locally or connect to the existing repo
repo = Repository(local_dir=repo_dir, clone_from=repo_id)

# Upload the model to the Hugging Face repository
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.pt",  # This is where your model will be saved in the repo
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model successfully uploaded to {repo_url}")
# %%
from huggingface_hub import hf_hub_download
import torch

# Define your Hugging Face repo ID
repo_id = "rievil/crackenpy"  # Update this to your repo if different

# Define the model filename in the repository
filename = "model.pt"

# Download the model from the Hugging Face Hub
model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="/Users/richarddvorak/Documents/Misc",
)

# Load the model into PyTorch
model = torch.load(model_path)

# Set the model to evaluation mode if necessary
model.eval()

print("Model successfully downloaded and loaded.")

# %%
from huggingface_hub import HfApi

# Define your Hugging Face repo ID
repo_id = "rievil/crackenpy"  # Update this if different

# Initialize the Hugging Face API
api = HfApi()

# Get the model metadata
model_info = api.model_info(repo_id)

# Print the download count
print(f"Model '{repo_id}' has been downloaded {model_info.downloads} times.")
