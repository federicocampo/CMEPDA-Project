def convert_to_pgm(fname, dest_folder):
  ''' This function converts .png images into
      .pgm images. This is done to easen the
      reading process for the CNN and to make
      it more universal.

      Parameters:
        - fname = name of the image to convert
        - dest_folder = folder path to which the image will be saved
  '''

  # If the desired folder to which save the new
  # images doesn't exist, here it's created.
  
  if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
    
  # Change the name of the filename to pgm extension
  dest_fname = os.path.basename(fname).replace('.png', '.pgm')
  # Define the destination folder path
  dest_fname = os.path.join(dest_folder, dest_fname)
  # Convert the image to grayscale and save it
  Image.open(fname).convert('L').save(dest_fname)

import os
from PIL import Image
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Tool to convert .png images into .pgm images contained in the same folder")
parser.add_argument('-path', help='You need to give me the path pointing to the folder containing Test_png and Train_png folders', type=str)
args = parser.parse_args()

PATH = args.path

# For Train_png and Test_png containing 0 and 1 labelled png images convert every image in pgm
for data_path in [os.path.join(PATH, "Train_png"), os.path.join(PATH, "Test_png")]:
    for path, folders, fnames in os.walk(data_path):
        # Using convert_to_pgm function for every image
        for fname in fnames:
            abs_path = os.path.join(path, fname)
                
            # New images are saved in the same general folder but are now contained
            # in the Train and Test folders instead of Train_png and Test_png.
            dest_folder = path.replace('Train_png', 'Train').replace('Test_png', 'Test')
            convert_to_pgm(abs_path, dest_folder)