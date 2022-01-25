# Importing libraries
import time
from selenium import webdriver
# Importing functions
from src.features.functions import creating_url, creating_image_folder, getting_url, finding_image


driver = webdriver.Chrome()

def scraping_images():
    glass_type, search_url = creating_url()
    folder_name = creating_image_folder(glass_type)
    getting_url(search_url)
    n = finding_image(folder_name)
    return n

def scraping_finish(n):
    print("\n\n\n")
    time.sleep(10)
    driver.quit()
    print('LOADING IS COMPLETED!')
    print()
    print('Number of downloaded images: ', n)


def main():
    n = scraping_images()
    scraping_finish(n)

if __name__ == "__main__":
    main()