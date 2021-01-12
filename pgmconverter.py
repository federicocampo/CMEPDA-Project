def convert_to_pgm(fname, dest_folder):
  # Create a new folder to put converted images in
  if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
    
  # Change the name of the filename to png extension
  dest_fname = os.path.basename(fname).replace('.png', '.pgm')
  # Define the destination file path
  dest_fname = os.path.join(dest_folder, dest_fname)
  # Convert the image to grayscale and save it
  Image.open(fname).convert('L').save(dest_fname)

import os
from PIL import Image

general_path = r"C:\Users\feder\Desktop"

folder_list = ['db2_3levels_nodenoise', 'db2_3levels_yesdenoise',
                'sym2_3levels_nodenoise', 'sym2_3levels_yesdenoise',
                'db5_4levels_nodenoise', 'db5_4levels_yesdenoise']

for i, folder in enumerate(folder_list):
    PATH = os.path.join(general_path, folder)

    for data_path in [os.path.join(PATH, "Train_png"), os.path.join(PATH, "Test_png")]:
        for path, folders, fnames in os.walk(data_path):
            #Using convert_to_png function to every filename
            for fname in fnames:
                abs_path = os.path.join(path, fname)
                #Create folders for png images at the same path of original pgm images

                dest_folder = path.replace('Train_png', 'Train').replace('Test_png', 'Test')
                convert_to_pgm(abs_path, dest_folder)