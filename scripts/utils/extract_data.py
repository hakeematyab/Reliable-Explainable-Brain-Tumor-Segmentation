import os
import tarfile


def extract_tar_file(tar_file_path, extract_path="./data", fraction=0.25):
    """
    Extracts a fraction of the total files from a tar archive.

    Args:
        tar_file_path (str): Path to the tar file.
        extract_path (str): Directory where selected files will be extracted.
        fraction (float): Fraction of files to extract (e.g., 0.25 for 1/4).

    Returns:
        None
    """
    if not os.path.exists(tar_file_path):
        raise FileNotFoundError(f"Tar file not found: {tar_file_path}")

    os.makedirs(extract_path, exist_ok=True)

    with tarfile.open(tar_file_path, "r:*") as tar:
        members = tar.getmembers()
        total_files = len(members)
        num_to_extract = max(1, int(total_files * fraction))

        print(f"Total files in archive: {total_files}")
        print(f"Extracting {num_to_extract} files ({fraction*100:.2f}% of total)...")

        selected_files = members[:num_to_extract+1]
        tar.extractall(path=extract_path, members=selected_files)

    print(f"Extracted {num_to_extract} files to '{extract_path}'")

if __name__ == "__main__":
    """
    Enter the tar file path and the destination path.
    """
    tar_file_path = "/home/hakeem.at/.cache/kagglehub/datasets/dschettler8845/brats-2021-task1/versions/1/BraTS2021_Training_Data.tar"
    destination_path = "./data/brats2021"
    fraction = 0.3
    try:
        extract_tar_file(tar_file_path, destination_path, fraction)
    except Exception as e:
        print(f"An error occurred: {e}")
