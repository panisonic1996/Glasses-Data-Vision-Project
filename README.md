#  Bar Glasses Recognition 
# :cocktail: :clinking_glasses: :wine_glass: :tropical_drink: :tumbler_glass:

This is a project of recognizing the type of bar glass from a picture.

### Project steps:

  1). Images were downloaded from Google using Python and Selenium Webdriver, placed into separate folders.
  
  *folder 'image_scraping'*
  
  2). Images were selected, cropped, unnecessary removed, changed to square shape.
  
  3). Test, validation and train datasets were organized, images renamed.
  
   *folder 'dataset_organization'*
  
  4). Image classifier (convolutional neural network, functional way, using  tensorflow, keras) was built.
  
   *folder 'model_building'*
  
  5). Model was evaluated, confusion was plotted.
  
  *folder 'model_building'*
          
        Results:
      - train dataset: loss 0.0513, accuracy: 0.9791
      - validation dataset: loss 0.1250, accuracy: 0.9677
       -test dataset: loss 0.1584, accuracy: 0.9624

*Link to raw and processed datasets:*
https://drive.google.com/drive/folders/1D79ZgPXy0enkB9CVQcxB60N36szftUAj?usp=sharing

**Technologies:**

 Language: Python;
 
 Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, PIL, selenium, requests, os, shutil.
