import os
from src.features.constants import image_folder_processed, dataset_processed, train_dataset, test_dataset, test_val_dataset, val_dataset
from src.features.functions import making_image_square, splitting_data, changing_images_name
import shutil


# Making images square
def changing_images(path):
    for subdir, dirs, images in os.walk(path):
        for image in images:
            # Image name
            image_path = os.path.join(subdir, image)
            making_image_square(image_path)
    print('CHANGING HAS BEEN FINISHED!')


# Splitting images to train/test/validation folders
def dataset_organization():
    # Splitting to train and test_val folders
    split_size = 0.4
    splitting_data(image_folder_processed, train_dataset, test_val_dataset, split_size)
    # Splitting to test and val folders
    split_size = 0.5
    splitting_data(test_val_dataset, test_dataset, val_dataset, split_size)
    shutil.rmtree(test_val_dataset, ignore_errors=True)
    shutil.rmtree(image_folder_processed, ignore_errors=True)
    print('FOLDERS HAVE BEEN REMOVED!')


def main():
    path = image_folder_processed
    changing_images(path)
    dataset_organization()
    changing_images_name(dataset_processed)


if __name__ == "__main__":
    main()