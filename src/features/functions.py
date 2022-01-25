import time
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from PIL import Image
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


# 1). Functions for scraping images
# Importing constants
from src.features.constants import google_image, image_folder_raw, driver, button_more_results, container_XPath, full_image_XPath

def creating_url():
    # ask for a user input
    glass_type = input('What type of glass are you looking for? -  ')
    # get url query string
    search_url = google_image.format(q=glass_type)
    return glass_type, search_url

def creating_image_folder(glass_type):
    # creating a directory to save images
    os.chdir(image_folder_raw)
    folder_name = glass_type
    print('Image folder created!')
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    return folder_name


def downloading_image(image_url, folder_name, number):
    # write image to file
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, str(number) + ".jpg"), 'wb') as file:
            file.write(response.content)

def getting_url(search_url):
    driver.get(search_url)
    # Scrolling web page down
    last_height = driver.execute_script('return document.body.scrollHeight')
    while True:
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        time.sleep(2)
        new_height = driver.execute_script('return document.body.scrollHeight')
        if new_height == last_height:
            # Clicking button 'Show more results'
            try:
                driver.find_element(By.XPATH, button_more_results).click()
                print('Button clicked')
                time.sleep(5)
            except:
                break
        last_height = new_height
    print('Page scrolled down!')
    time.sleep(2)
    # Scrolling web page up
    driver.execute_script("window.scrollTo(0, 0);")
    print('Page scrolled up!')

def finding_image(folder_name):
    # Find containers with images
    containers = driver.find_elements(By.XPATH, container_XPath)
    wait = WebDriverWait(driver, 10)
    images_count = len(containers)
    # Number of clicked container
    i = 0
    # Number of downloaded images
    n = 0
    for image_preview in containers:
        # Clicking container with image
        try:
            print("\n\n\n")
            image_preview_url = image_preview.get_attribute("src")
            image_preview_name = image_preview.get_attribute("alt")
            print('Preview Image Name:  ', image_preview_name)
            try:
                image_preview.click()
            except:
                image_preview.click()
            i += 1
            # Getting link  to high resolution image
            try:
                image_high_resolution = wait.until(EC.presence_of_element_located((By.XPATH, full_image_XPath)))
                image_url = image_high_resolution.get_attribute('src')
                image_name = image_high_resolution.get_attribute('alt')
                # Error handling
                startTime = time.time()
                while (image_url.startswith("data") or image_url.startswith("https://encrypted")) \
                        and time.time() - startTime < 3:
                    image_url = image_high_resolution.get_attribute('src')

                if image_name != image_preview_name:
                    errorMsg = """Full image name not equals to preview image name:\n Preview Image Name:  %s;\n Full Image Name:  %s""" % (
                        image_preview_name,
                        image_name
                    )
                    print(errorMsg)
                    raise Exception(errorMsg)
            except:
                print("Something went wrong while getting full image url for downloading")
                continue

        except:
            print("Something went wrong while clicking the container with image!")
            continue

        # Downloading image
        try:
            downloading_image(image_url, folder_name, i)
            n += 1
            print("Downloaded element %s out of %s total." % (n, images_count + 1))
        except:
            print("Couldn't download an image %s, continuing downloading the next one" % i)
    return n


# 2). Functions for image preprocessing
def making_image_square(image_path):
    image = Image.open(image_path)
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    if(width != height):
        large_side = width if width > height else height
        background = Image.new('RGB', (large_side, large_side), (255, 255, 255, 255))
        offset = (int(round(((large_side - width) / 2), 0)), int(round(((large_side - height) / 2), 0)))
        background.paste(image, offset)
        background.save(image_path)
        print(f'Image {image_path} has been resized!')
    else:
        print(f'Image {image_path} is a square!')

# 3). Functions to organize dataset for training
# Splitting images to train/test/validation folders
def splitting_data(path_to_data, path_to_save_train, path_to_save_test_or_val, split_size):

    folders = os.listdir(path_to_data)

    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.jpg'))
        x_train, x_test_or_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:
            path_to_folder = os.path.join(path_to_save_train, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x, path_to_folder)

        for x in x_test_or_val:
            path_to_folder = os.path.join(path_to_save_test_or_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x, path_to_folder)

    print('Sorting images into folders is completed!')

# Renaming pictures
def changing_images_name(path):
    for root, dirs, images in os.walk(path):
        i = 1
        for image in images:
            new_file_name = "{}.jpg".format(i)
            os.rename(os.path.join(root, image), os.path.join(root, new_file_name))
            i = i+1
    print('Files renamed successfully!')


# 4). Image Generators
from src.features.constants import target_size, color_mode, classes

def creating_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True
    )
    preprocessor = ImageDataGenerator(
        rescale=1/255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        classes=classes,
        target_size=target_size,
        color_mode=color_mode,
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        classes=classes,
        target_size=target_size,
        color_mode=color_mode,
        shuffle=False,
        batch_size=batch_size,
    )

    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        classes=classes,
        target_size=target_size,
        color_mode=color_mode,
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator


# 5). Model architecture
def glasses_recognition(number_of_classes):

    my_input = Input(shape=(224, 224, 3))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(my_input)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(256, activation='relu')(x)
    Flatten(),
    x = Dense(number_of_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

# 6).  Model Plots
def model_history(history):
    # Plotting Accuracies
    plt.figure(figsize=(20,7))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Glasses Recognition Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    # Plotting Losses
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Glasses Recognition Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

def plotting_images(images_array):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.flatten()
    for img, ax in zip(images_array, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plotting_confusion_matrix(test_labels_names, predict_results):
    cm = confusion_matrix(test_labels_names, predict_results)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(17, 8))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.title('Confusion Matrix', fontsize=15, pad=15)
    plt.ylabel('Actual Glass Type', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Glass Type', fontsize=12, labelpad=10)
    plt.show()

def single_image_prediction(model, imgpath):
    image = tf.io.read_file(imgpath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, target_size)
    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predictions = np.argmax(predictions)
    return predictions


