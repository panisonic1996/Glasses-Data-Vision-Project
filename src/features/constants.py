from selenium import webdriver

driver = webdriver.Chrome()

# Web page elements
google_image = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
button_more_results = '//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input'
container_XPath = """//*[@id="islrg"]/div[1]/div/a[1]/div[1]/img"""
full_image_XPath = """//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img"""

# Folder paths
image_folder_raw = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\raw'
image_folder_processed = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\processed'
dataset_processed = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\dataset(processed)'
train_dataset = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\dataset(processed)\\train'
test_dataset = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\dataset(processed)\\test'
val_dataset = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\dataset(processed)\\val'
test_val_dataset = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\data(images)\\dataset(processed)\\test+val'
model_saving = 'C:\\Datasets and Projects\\GLASSES RECOGNITION\\models'

# Model elements
number_classes = 9
batch_size = 8
target_size = (224, 224)
color_mode = "rgb"
classes = ['CHAMPAGNE FLUTE', 'HIGHBALL GLASS', 'HURRICANE GLASS', 'MARGARITA GLASS', 'MARTINI GLASS',
           'OLD FASHIONED GLASS', 'SNIFTER GLASS', 'STEAMLESS WINE GLASS', 'WINE GLASS']








