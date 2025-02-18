import os
import zipfile
import subprocess

def download_dataset(dataset_name="luchack/ucf101", destination="./dataset"):
    """
    Download the UCF101 dataset from Kaggle and place it in a 'dataset' folder.

    :param dataset_name: Kaggle dataset identifier.
    :param destination: Directory where to save the dataset, relative to the script's location.
    """
    # Ensure the API token is available
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        raise FileNotFoundError("Kaggle API credentials not found. Place kaggle.json in ~/.kaggle/ directory.")

    # Create the destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    # Download the dataset
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", destination], check=True)

    # Find the downloaded zip file
    zip_files = [f for f in os.listdir(destination) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError("No zip file found after download.")
    zip_file = os.path.join(destination, zip_files[0])

    # Unzip the dataset
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)
    
    # Remove the zip file after extraction
    os.remove(zip_file)
    print(f"Dataset '{dataset_name}' has been downloaded and unzipped to {destination}")

if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as e:
        print(f"An error occurred: {e}")