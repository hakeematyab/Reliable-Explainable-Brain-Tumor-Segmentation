import os
import json

def download_kaggle_dataset(dataset_name, download_path="./data"):
    """
    Downloads a dataset from Kaggle after prompting the user for an API key.

    Args:
        dataset_name (str): Kaggle dataset name in the format "owner/dataset-name".
        download_path (str): Directory to save the dataset (default: "./dataset").

    Returns:
        None
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json_path):
        api_key = input("Enter your Kaggle API key: ").strip()
        username = input("Enter your Kaggle username: ").strip()
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": api_key}, f)
        os.chmod(kaggle_json_path, 0o600)

    os.makedirs(download_path, exist_ok=True)
    import kaggle
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

if __name__ == "__main__":
    download_kaggle_dataset("dschettler8845/brats-2021-task1", "./data/brats2021")