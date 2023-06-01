"""
Utility file to download and extract datasets from google drive.
(data too big fail to download with l2l)
"""
import gdown
import os
import zipfile
import torchvision
from learn2learn.vision.datasets.cifarfs import CIFARFS


def download_zip(id, destination):
    url = "https://drive.google.com/uc?id="
    gdown.download(url + id, destination, quiet=False)


def proces_path(path):
    if not os.path.isabs(path):
        cwd = os.getcwd()
        path = os.path.join(cwd, path)
    return path


def download_and_extract_if_not_exists(id, path, name_dataset, delete_zip=False):
    """
    Download and extract a dataset if it does not exist in the destination folder.
    Note that if the extraction process is interrupted, the installation will fail.
    Only download trusted data sources.

    Args:
        id (_type_): drive id for the dataset
        path (str): path of the data (relative or absolute).
            Note that "/data" is absolute path (use "data" for relative path))
        name_dataset (str): name of the dataset
        delete_zip (bool): if true, delete the zip file after extraction (default: False)
    """
    path = proces_path(path)

    path_extract = os.path.join(path, name_dataset + "/")
    print(path_extract)
    if os.path.exists(path_extract) and os.path.isdir(path_extract):
        print(f"Dataset {name_dataset} already extracted")
        return

    os.mkdir(path_extract)
    zip_path = os.path.join(path, name_dataset + ".zip")
    if os.path.exists(zip_path):
        print(f"Dataset {name_dataset} already downloaded")
    else:
        # get current working directory and append relative path to data
        print("Downloading dataset: ", name_dataset)
        download_zip(id, zip_path)

    print("Extracting dataset: ", name_dataset)
    with zipfile.ZipFile(zip_path, "r") as zfile:
        zfile.extractall(path_extract)

    if delete_zip:
        os.remove(zip_path)

import learn2learn as l2l
DATASET_URLS = {
    "cifarfs": "1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI"
    }
DATASET_MODULE = {"cifarfs": CIFARFS}

def download_datasets(list_dataset_to_download, save_path = "datasets/"):
    for dataset_name in list_dataset_to_download:
        dataset_id = DATASET_URLS[dataset_name]
        download_and_extract_if_not_exists(
            dataset_id, save_path, dataset_name, delete_zip=False
        )
        dataset = DATASET_MODULE[dataset_name](root=save_path, transform = torchvision.transforms.ToTensor())

        # test the dataset (the l2l metadataset save additional infos)
        dataset = l2l.data.MetaDataset(dataset)
        transforms = [
            l2l.data.transforms.NWays(dataset, n=5),
            l2l.data.transforms.KShots(dataset, k=2),
            l2l.data.transforms.LoadData(dataset),
            ]
        taskset = l2l.data.TaskDataset(dataset, transforms, num_tasks=1)
        batch = next(taskset)
        print(batch)


if __name__ == "__main__":
    download_datasets(["cifarfs"])