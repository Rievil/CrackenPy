import os
import pkg_resources

import crackpy_models


ONLINE_CRACKPY_MODELS = {
    "0": "model1.pt",
    "1": "model2.pt",
}


def download_model(key):
    from huggingface_hub import hf_hub_download

    model = pkg_resources.resource_listdir("crackpy_models", "")
    online_models = ONLINE_CRACKPY_MODELS
    count = model.count(online_models[key])

    if count == 0:
        module_path = crackpy_models.__file__
        tar_folder = os.path.dirname(module_path)

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
    model = pkg_resources.resource_listdir("crackpy_models", "")
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