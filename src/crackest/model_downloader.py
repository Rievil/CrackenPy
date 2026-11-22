import os

import crackpy_models


ONLINE_CRACKPY_MODELS = {
    "0": "model1.pt",
    "1": "model2.pt",
}

# Encoder each online checkpoint was trained with
ONLINE_CRACKPY_ENCODERS = {
    "0": "resnext50_32x4d",
    "1": "resnext101_32x8d",
}


def models_folder():
    """Folder of the crackpy_models package, where checkpoints are stored."""
    return os.path.dirname(crackpy_models.__file__)


def download_model(key):
    from huggingface_hub import hf_hub_download

    model = os.listdir(models_folder())
    online_models = ONLINE_CRACKPY_MODELS
    count = model.count(online_models[key])

    if count == 0:
        tar_folder = models_folder()

        print(
            "Downloading deep learing model '{:s}' for module crackpy".format(
                online_models[key].replace(".pt", "")
            )
        )
        hf_hub_download(
            repo_id="rievil/crackenpy",
            filename=online_models[key],
            local_dir=tar_folder,
        )
        print("... done downloading")
        # gdown.download(id=url_id, output=out_file, quiet=False)


def update_models():
    model = os.listdir(models_folder())
    online_models = ONLINE_CRACKPY_MODELS

    count_d = 0
    for key in online_models:
        count = model.count(online_models[key])
        if count == 0:
            count_d += 1
            download_model(key)

    if count_d == 0:
        print("All models are already downloaded")
    else:
        print("Downloaded {:d} models".format(count_d))