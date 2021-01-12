def dwtcoefftoarray(myim, wavelet, level, denoise):
    """ This function collects all the coefficients of the DWT and converts them
        in a flat array which can be passed to the Deep Neural Network.
    """
    if denoise == 'yes':
        myimfl = img_as_float(myim)
        myim = denoise_wavelet(myimfl, method='BayesShrink', mode='soft', rescale_sigma='True')
    elif denoise == 'no':
        pass

    coeffs = pywt.wavedec2(myim, wavelet, level=level)
    coeffsarray = pywt.ravel_coeffs(coeffs)
    return coeffsarray


def dwtanalysis(myim, wavelet, level, denoise):
    """ This function decomposes the original image with a Discrete Wavelet Transformation
        using the desired wavelet family up to the fifth level. It keeps all the details
        coefficients and mask the resulted approximated image in order to enhance the
        visibility of all the details. One can choose to denoise the original image
        prior to the DWT decomposition and reconstruction.
    """
    if denoise == 'yes':
        myimfl = img_as_float(myim)
        myim = denoise_wavelet(myimfl, method='BayesShrink', mode='soft', rescale_sigma='True')
    elif denoise == 'no':
        pass

    if level == 2:
        cA, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    elif level == 3:
        cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    elif level == 4:
        cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    elif level == 5:
        cA, (cH5, cV5, cD5), (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    else:
        pass

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

    if level == 3 or level == 4:
        std30 = np.std(cH3)
        std31 = np.std(cV3)
        std32 = np.std(cD3)

        ncH3 = pywt.threshold(cH3, std30, mode = 'hard', substitute = 0.)
        ncV3 = pywt.threshold(cV3, std31, mode = 'hard', substitute = 0.)
        ncD3 = pywt.threshold(cD3, std32, mode = 'hard', substitute = 0.)
    else:
        pass

    if level == 4 or level == 5:
        std40 = np.std(cH4)
        std41 = np.std(cV4)
        std42 = np.std(cD4)

        ncH4 = pywt.threshold(cH4, std40, mode = 'hard', substitute = 0.)
        ncV4 = pywt.threshold(cV4, std41, mode = 'hard', substitute = 0.)
        ncD4 = pywt.threshold(cD4, std42, mode = 'hard', substitute = 0.)
    else:
        pass

    if level == 5:
        std50 = np.std(cH5)
        std51 = np.std(cV5)
        std52 = np.std(cD5)

        ncH5 = pywt.threshold(cH5, std50, mode = 'hard', substitute = 0.)
        ncV5 = pywt.threshold(cV5, std51, mode = 'hard', substitute = 0.)
        ncD5 = pywt.threshold(cD5, std52, mode = 'hard', substitute = 0.)
    else:
        pass

    """ To let things be more readable I define new_coeff,
        this is just so that waverec2 (the function needed to reconstruct
        the image from a set of given coefficient) can do what it does.
    """
    if level == 2:
        new_coeff = ncA, (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 
    elif level == 3:
        new_coeff = ncA, (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 
    elif level == 4:
        new_coeff = ncA, (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)     
    elif level == 5:
        new_coeff = ncA, (ncH5, ncV5, ncD5), (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 

    mynewim = pywt.waverec2(new_coeff, wavelet)
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
from skimage.restoration import denoise_wavelet
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

level = 4
denoise = 'no'

""" Getting image path to open the image
"""
general_path = r"C:\Users\feder\Desktop\Computational Methods for Experimental Physics and Data Analysis\IMAGES\Mammography_micro"

for k, folder in enumerate(['Train', 'Test']):
    zero_images_path = os. path.join(general_path, folder, '0')
    one_images_path = os. path.join(general_path, folder, '1')

    zero_images_names = glob.glob(os.path.join(zero_images_path, '*.pgm'))
    one_images_names = glob.glob(os.path.join(one_images_path, '*.pgm'))

    save_folder = f'{folder}_png'

    for j, wavelet_type in enumerate(['db5']):
        for i, image_path in enumerate(zero_images_names):
            im = Image.open(image_path)
            myim, mynewim = dwtanalysis(im, wavelet_type, level=level, denoise=denoise)

            save_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', f'{wavelet_type}_{level}levels_{denoise}denoise', save_folder, '0')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            final_path = os.path.join(save_path, f'{i}.png')

            mynewim = mynewim.astype(np.uint8)
            mynewim = Image.fromarray(mynewim)
            mynewim.save(final_path)

        for i, image_path in enumerate(one_images_names):
            im = Image.open(image_path)
            myim, mynewim = dwtanalysis(im, wavelet_type, level=level, denoise=denoise)

            save_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', f'{wavelet_type}_{level}levels_{denoise}denoise', save_folder, '1')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            final_path = os.path.join(save_path, f'{i}.png')

            mynewim = mynewim.astype(np.uint8)
            mynewim = Image.fromarray(mynewim)
            mynewim.save(final_path)