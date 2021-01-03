from PIL import Image
import pywt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

""" Getting image path to open the image
"""
image_path = r"C:\Users\feder\Desktop\Computational Methods for Experimental Physics and Data Analysis\cmepda_medphys\L4_code\0039t1_2_1_1.pgm"
myim = Image.open(image_path)

""" Choosing the wavelet family and the level of the decomposition
"""
wavelet = 'sym3'
level = 3
cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level) # Here I get my approximated image and the relative coefficients
# (cH5, cV5, cD5), (cH4, cV4, cD4), 


""" Now, I get the standard deviation for each matrix (image and coefficients).
    The std will act as a treshold so that if abs(value) < 0. --> value = 0.
                                           elif abs(value) > 0. --> value = value
"""
std00 = np.std(cA)
ncA = pywt.threshold(cA, std00, mode = 'hard', substitute = 0.)

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

# Fourth and fifth levels can be implemented from here
"""std40 = np.std(cH4)
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
ncD5 = pywt.threshold(cD5, std52, mode = 'hard', substitute = 0.)"""

""" To let things be more readable I define new_coeff,
    this is just so that waverec2 (the function needed to reconstruct
    the image from a set of given coefficient) can do what it does.
"""
new_coeff = ncA, (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)
# (ncH5, ncV5, ncD5), (ncH4, ncV4, ncD4), 

mynewim = pywt.waverec2(new_coeff, wavelet)

mynewim2 = pywt.threshold(mynewim, 0., mode = 'greater', substitute = 0.)

""" In this plot is shown a comparison between the modified images and the original one-
"""
plt.figure(1)
plt.subplot(1,3,1)
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

plt.subplot(1,3,2)
plt.title("Modificata")
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

plt.subplot(1,3,3)
plt.title("Modificata 2")
plt.imshow(mynewim2, cmap='gray')
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,      
    top=False, 
    left=False,
    right=False,        
    labelbottom=False,
    labelleft=False)

plt.show()