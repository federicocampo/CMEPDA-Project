def threelvldecomp(im_path, wavelet):
    """ This function decompose the original image with a Discrete Wavelet Transformation
        using the desired wavelet family up to the third level. It keeps all the details
        coefficient and mask the resulted approximated image in order to enhance the
        visibility of all the details.
    """
    myim = Image.open(im_path)

    wavelet = wavelet
    level = 3
    mode = 'periodization'

    cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, mode=mode, level=level) # Here I get my approximated image and the relative coefficients

    """ Now, I get the standard deviation for each matrix (image and coefficients).
        The std will act as a treshold so that if abs(value) < 0. --> value = 0.
                                            elif abs(value) > 0. --> value = value
    """
    ncA = np.zeros_like(cA)

    std10 = np.std(cH1)
    std11 = np.std(cV1)
    std12 = np.std(cD1)

    ncH1 = pywt.threshold(cH1, std10, mode = 'hard', substitute = 0.)
    ncV1 = pywt.threshold(cV1, std11, mode = 'hard', substitute = 0.)
    ncD1 = pywt.threshold(cD1, std12, mode = 'hard', substitute = 0.)

    std20 = np.std(cH2)
    std21 = np.std(cV2)
    std22 = np.std(cD2)

    ncH2 = pywt.threshold(cH2, std20, mode = 'hard', substitute = 0.)
    ncV2 = pywt.threshold(cV2, std21, mode = 'hard', substitute = 0.)
    ncD2 = pywt.threshold(cD2, std22, mode = 'hard', substitute = 0.)

    std30 = np.std(cH3)
    std31 = np.std(cV3)
    std32 = np.std(cD3)

    ncH3 = pywt.threshold(cH3, std30, mode = 'hard', substitute = 0.)
    ncV3 = pywt.threshold(cV3, std31, mode = 'hard', substitute = 0.)
    ncD3 = pywt.threshold(cD3, std32, mode = 'hard', substitute = 0.)

    """ To let things be more readable I define new_coeff,
        this is just so that waverec2 (the function needed to reconstruct
        the image from a set of given coefficient) can do what it does.
    """

    new_coeff = ncA, (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 

    mynewim = pywt.waverec2(new_coeff, wavelet, mode=mode)
    mynewim = pywt.threshold(mynewim, 0., mode = 'greater', substitute = 0.)

    return myim, mynewim


def fourlvldecomp(im_path, wavelet):
    """ This function decompose the original image with a Discrete Wavelet Transformation
        using the desired wavelet family up to the fourth level. It keeps all the details
        coefficient and mask the resulted approximated image in order to enhance the
        visibility of all the details.
    """
    myim = Image.open(im_path)

    wavelet = wavelet
    level = 4
    mode = 'periodization'

    cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, mode=mode, level=level) # Here I get my approximated image and the relative coefficients

    """ Now, I get the standard deviation for each matrix (image and coefficients).
        The std will act as a treshold so that if abs(value) < 0. --> value = 0.
                                            elif abs(value) > 0. --> value = value
    """
    ncA = np.zeros_like(cA)

    std10 = np.std(cH1)
    std11 = np.std(cV1)
    std12 = np.std(cD1)

    ncH1 = pywt.threshold(cH1, std10, mode = 'hard', substitute = 0.)
    ncV1 = pywt.threshold(cV1, std11, mode = 'hard', substitute = 0.)
    ncD1 = pywt.threshold(cD1, std12, mode = 'hard', substitute = 0.)

    std20 = np.std(cH2)
    std21 = np.std(cV2)
    std22 = np.std(cD2)

    ncH2 = pywt.threshold(cH2, std20, mode = 'hard', substitute = 0.)
    ncV2 = pywt.threshold(cV2, std21, mode = 'hard', substitute = 0.)
    ncD2 = pywt.threshold(cD2, std22, mode = 'hard', substitute = 0.)

    std30 = np.std(cH3)
    std31 = np.std(cV3)
    std32 = np.std(cD3)

    ncH3 = pywt.threshold(cH3, std30, mode = 'hard', substitute = 0.)
    ncV3 = pywt.threshold(cV3, std31, mode = 'hard', substitute = 0.)
    ncD3 = pywt.threshold(cD3, std32, mode = 'hard', substitute = 0.)

    std40 = np.std(cH4)
    std41 = np.std(cV4)
    std42 = np.std(cD4)

    ncH4 = pywt.threshold(cH4, std40, mode = 'hard', substitute = 0.)
    ncV4 = pywt.threshold(cV4, std41, mode = 'hard', substitute = 0.)
    ncD4 = pywt.threshold(cD4, std42, mode = 'hard', substitute = 0.)

    """ To let things be more readable I define new_coeff,
        this is just so that waverec2 (the function needed to reconstruct
        the image from a set of given coefficient) can do what it does.
    """

    new_coeff = ncA, (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 

    mynewim = pywt.waverec2(new_coeff, wavelet, mode=mode)
    mynewim = pywt.threshold(mynewim, 0., mode = 'greater', substitute = 0.)

    return myim, mynewim


def fivelvldecomp(im_path, wavelet):
    """ This function decompose the original image with a Discrete Wavelet Transformation
        using the desired wavelet family up to the fifth level. It keeps all the details
        coefficient and mask the resulted approximated image in order to enhance the
        visibility of all the details.
    """
    myim = Image.open(im_path)

    wavelet = wavelet
    level = 5
    mode = 'periodization'

    cA, (cH5, cV5, cD5), (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, mode=mode, level=level) # Here I get my approximated image and the relative coefficients

    """ Now, I get the standard deviation for each matrix (image and coefficients).
        The std will act as a treshold so that if abs(value) < 0. --> value = 0.
                                            elif abs(value) > 0. --> value = value
    """
    ncA = np.zeros_like(cA)

    std10 = np.std(cH1)
    std11 = np.std(cV1)
    std12 = np.std(cD1)

    ncH1 = pywt.threshold(cH1, std10, mode = 'hard', substitute = 0.)
    ncV1 = pywt.threshold(cV1, std11, mode = 'hard', substitute = 0.)
    ncD1 = pywt.threshold(cD1, std12, mode = 'hard', substitute = 0.)

    std20 = np.std(cH2)
    std21 = np.std(cV2)
    std22 = np.std(cD2)

    ncH2 = pywt.threshold(cH2, std20, mode = 'hard', substitute = 0.)
    ncV2 = pywt.threshold(cV2, std21, mode = 'hard', substitute = 0.)
    ncD2 = pywt.threshold(cD2, std22, mode = 'hard', substitute = 0.)

    std30 = np.std(cH3)
    std31 = np.std(cV3)
    std32 = np.std(cD3)

    ncH3 = pywt.threshold(cH3, std30, mode = 'hard', substitute = 0.)
    ncV3 = pywt.threshold(cV3, std31, mode = 'hard', substitute = 0.)
    ncD3 = pywt.threshold(cD3, std32, mode = 'hard', substitute = 0.)

    std40 = np.std(cH4)
    std41 = np.std(cV4)
    std42 = np.std(cD4)

    ncH4 = pywt.threshold(cH4, std40, mode = 'hard', substitute = 0.)
    ncV4 = pywt.threshold(cV4, std41, mode = 'hard', substitute = 0.)
    ncD4 = pywt.threshold(cD4, std42, mode = 'hard', substitute = 0.)

    std50 = np.std(cH5)
    std51 = np.std(cV5)
    std52 = np.std(cD5)

    ncH5 = pywt.threshold(cH5, std50, mode = 'hard', substitute = 0.)
    ncV5 = pywt.threshold(cV5, std51, mode = 'hard', substitute = 0.)
    ncD5 = pywt.threshold(cD5, std52, mode = 'hard', substitute = 0.)

    """ To let things be more readable I define new_coeff,
        this is just so that waverec2 (the function needed to reconstruct
        the image from a set of given coefficient) can do what it does.
    """

    new_coeff = ncA, (ncH5, ncV5, ncD5), (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 

    mynewim = pywt.waverec2(new_coeff, wavelet, mode=mode)
    mynewim = pywt.threshold(mynewim, 0., mode = 'greater', substitute = 0.)

    return myim, mynewim


def savecomparison(myim, mynewim, save_path, title_of_image, name_of_image):
    """ This function saves an image showing the comparison between the original image and the
        reconstructed one in which details are highly enhanced.
    """

    dpi = 96

    plt.figure(figsize=(1200/dpi, 600/dpi), dpi=dpi)
    plt.subplot(1,2,1)
    plt.title("Normale")
    plt.imshow(myim, cmap='gray')
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,      
        top=False, 
        left=False,
        right=False,        
        labelbottom=False,
        labelleft=False)
    plt.colorbar(orientation='horizontal')

    plt.subplot(1,2,2)
    plt.title(title_of_image)
    plt.imshow(mynewim, cmap='gray')
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,      
        top=False, 
        left=False,
        right=False,        
        labelbottom=False,
        labelleft=False)
    plt.colorbar(orientation='horizontal')

    final_path = os.path.join(save_path, name_of_image)

    plt.savefig(final_path, bbox_inches='tight')




from PIL import Image
import pywt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

""" Getting image path to open the image
"""

# BISOGNA MODIFICARE QUESTO PATH PER RICAVARE IL CORRETTO PATH DELL'IMMAGINE
image_path = r"C:\Users\feder\Desktop\Computational Methods for Experimental Physics and Data Analysis\cmepda_medphys\L4_code\0039t1_2_1_1.pgm"

for i, wavelet_type in enumerate(['haar', 'sym2', 'sym3', 'db2', 'db3']):
    # Here I do the magic with pywavelets
    myim, mynewim = threelvldecomp(image_path, wavelet_type)

    # The image will be saved in a the 'Comparison' folder on the Desktop of the User
    save_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Comparisons')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_name = f"threelevelswith{wavelet_type}.png"
    image_title = f"Altered with {wavelet_type}-3lvls"
    savecomparison(myim, mynewim, save_path, image_title, image_name)

for i, wavelet_type in enumerate(['haar', 'sym2', 'sym3', 'db2', 'db3']):
    # Here I do the magic with pywavelets
    myim, mynewim = fourlvldecomp(image_path, wavelet_type)

    # The image will be saved in a the 'Comparison' folder on the Desktop of the User
    save_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Comparisons')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_name = f"fourlevelswith{wavelet_type}.png"
    image_title = f"Altered with {wavelet_type}-4lvls"
    savecomparison(myim, mynewim, save_path, image_title, image_name)

for i, wavelet_type in enumerate(['haar', 'sym2', 'sym3', 'db2', 'db3']):
    # Here I do the magic with pywavelets
    myim, mynewim = fivelvldecomp(image_path, wavelet_type)

    # The image will be saved in a the 'Comparison' folder on the Desktop of the User
    save_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Comparisons')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_name = f"fivelevelswith{wavelet_type}.png"
    image_title = f"Altered with {wavelet_type}-5lvls"
    savecomparison(myim, mynewim, save_path, image_title, image_name)