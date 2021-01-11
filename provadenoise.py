from skimage.restoration import denoise_wavelet
from skimage import img_as_float
from matplotlib import pyplot
from PIL import Image
import pywt
import numpy

def threelvldecomp(myim, wavelet):
    """ This function decompose the original image with a Discrete Wavelet Transformation
        using the desired wavelet family up to the third level. It keeps all the details
        coefficient and mask the resulted approximated image in order to enhance the
        visibility of all the details.
    """

    level = 3

    cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level) # Here I get my approximated image and the relative coefficients

    """ Now, I get the standard deviation for each matrix (image and coefficients).
        The std will act as a treshold so that if abs(value) < 0. --> value = 0.
                                             elif abs(value) > 0. --> value = value
    """ 
    mult_val = 1.

    ncA = numpy.zeros_like(cA)

    std10 = numpy.std(cH1)*mult_val
    std11 = numpy.std(cV1)*mult_val
    std12 = numpy.std(cD1)*mult_val

    ncH1 = pywt.threshold(cH1, std10, mode = 'hard', substitute = 0.)
    ncV1 = pywt.threshold(cV1, std11, mode = 'hard', substitute = 0.)
    ncD1 = pywt.threshold(cD1, std12, mode = 'hard', substitute = 0.)

    std20 = numpy.std(cH2)*mult_val
    std21 = numpy.std(cV2)*mult_val
    std22 = numpy.std(cD2)*mult_val

    ncH2 = pywt.threshold(cH2, std20, mode = 'hard', substitute = 0.)
    ncV2 = pywt.threshold(cV2, std21, mode = 'hard', substitute = 0.)
    ncD2 = pywt.threshold(cD2, std22, mode = 'hard', substitute = 0.)

    std30 = numpy.std(cH3)*mult_val
    std31 = numpy.std(cV3)*mult_val
    std32 = numpy.std(cD3)*mult_val

    ncH3 = pywt.threshold(cH3, std30, mode = 'hard', substitute = 0.)
    ncV3 = pywt.threshold(cV3, std31, mode = 'hard', substitute = 0.)
    ncD3 = pywt.threshold(cD3, std32, mode = 'hard', substitute = 0.)

    """ To let things be more readable I define new_coeff,
        this is just so that waverec2 (the function needed to reconstruct
        the image from a set of given coefficient) can do what it does.
    """

    new_coeff = ncA, (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 

    mynewim = pywt.waverec2(new_coeff, wavelet)
    mynewim = pywt.threshold(mynewim, 0., mode = 'greater', substitute = 0.)

    return myim, mynewim

image_path = r"C:\Users\feder\Desktop\Computational Methods for Experimental Physics and Data Analysis\IMAGES\Mammography_micro\Train\0\0002s1_2_0.pgm_1.pgm"
myim = Image.open(image_path)
myimfl = img_as_float(myim)

myim2 = denoise_wavelet(myimfl, method='BayesShrink', mode='soft', rescale_sigma='True')

myim, contrastim = threelvldecomp(myim, 'sym2')
myim2, contrastim2 = threelvldecomp(myim2, 'sym2')

dpi = 96

pyplot.figure(1, figsize=(1200/dpi, 600/dpi), dpi=dpi)
pyplot.subplot(1,2,1)
pyplot.title("Normale")
pyplot.imshow(myim, cmap='gray')
pyplot.tick_params(
    axis='both',
    which='both',
    bottom=False,      
    top=False, 
    left=False,
    right=False,        
    labelbottom=False,
    labelleft=False)
pyplot.colorbar(orientation='horizontal')

pyplot.subplot(1,2,2)
pyplot.title("Normale")
pyplot.imshow(contrastim, cmap='gray')
pyplot.tick_params(
    axis='both',
    which='both',
    bottom=False,      
    top=False, 
    left=False,
    right=False,        
    labelbottom=False,
    labelleft=False)
pyplot.colorbar(orientation='horizontal')

pyplot.figure(2, figsize=(1200/dpi, 600/dpi), dpi=dpi)
pyplot.subplot(1,2,1)
pyplot.title("Denoised")
pyplot.imshow(myim2, cmap='gray')
pyplot.tick_params(
    axis='both',
    which='both',
    bottom=False,      
    top=False, 
    left=False,
    right=False,        
    labelbottom=False,
    labelleft=False)
pyplot.colorbar(orientation='horizontal')

pyplot.subplot(1,2,2)
pyplot.title("Denoised")
pyplot.imshow(contrastim2, cmap='gray')
pyplot.tick_params(
    axis='both',
    which='both',
    bottom=False,      
    top=False, 
    left=False,
    right=False,        
    labelbottom=False,
    labelleft=False)
pyplot.colorbar(orientation='horizontal')

pyplot.show()