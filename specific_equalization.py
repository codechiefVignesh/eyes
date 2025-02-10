import numpy as np
import matplotlib.pyplot as plt
import math, copy
import cv2 as cv
from skimage.exposure import match_histograms

# to round off the value as per rounding off rules of si system
def custom_round(value):
    diff = value - math.floor(value)
    if(diff >= 0.5):
        if(math.ceil(value) > 255):
            return 255
        return math.ceil(value)
    else:
        return math.floor(value)

# binary search method to find the greatest integer less than or equal to the value in an array
def search(value, array):
    low = 0
    high = len(array) - 1

    while(low <= high):
        mid = low + (high - low) // 2
        
        if(value == array[mid]):
            return mid
        elif(value < array[mid]):
            high = mid - 1
        else:
            low = mid + 1
    return high

# histogram equalisation method 
def histogram_equalisation(dark_dict, dark_img):
    pdf_dark = {k: 255 * sum(list(dark_dict.values())[:k]) for k in range(256)}
    equalised_img = copy.deepcopy(dark_img)

    for i in range(equalised_img.shape[0]):
        for j in range(equalised_img.shape[1]):
            equalised_img[i, j] = pdf_dark[equalised_img[i, j]]
    return equalised_img, pdf_dark

#histogram specification (matching) method
def histogram_matching(dark_dict, bright_dict, dark_img, bright_img):
    dark_pdf = {k:custom_round(255 * sum(list(dark_dict.values())[:k])) for k in range(256)}
    bright_pdf = {k:custom_round(255 * sum(list(bright_dict.values())[:k])) for k in range(256)}

    match_table = {i:0 for i in list(dark_dict.keys())}

    for i in range(256):
        j = search(list(dark_pdf.values())[i], list(bright_pdf.values()))
        match_table[list(dark_pdf.keys())[i]] = list(bright_pdf.keys())[j]

    specific_img = copy.deepcopy(dark_img)

    for i in range(specific_img.shape[0]):
        for j in range(specific_img.shape[1]):
            specific_img[i, j] = match_table[specific_img[i, j]]
    return specific_img, match_table   

def builtin_processing(dark_img, bright_img, equalised_img, specific_img):
    equalized_img = cv.equalizeHist(dark_img)
    matched_img = match_histograms(dark_img, bright_img)
    plt.figure()
    plt.subplot(121);plt.imshow(equalized_img, cmap = "grey"); plt.title("Built Equalised Image")
    plt.subplot(122); plt.imshow(matched_img, cmap = "grey"); plt.title("Built Specific Image")

    error_equal = sum((equalized_img.flatten() - equalised_img.flatten()) / len(equalised_img.flatten()))
    error_specific = sum((matched_img.flatten() - matched_img.flatten()) / len(matched_img.flatten()))
    return error_equal, error_specific

if __name__ == '__main__':

    # image reading 
    dark_img = cv.imread('pout-dark.jpg', 0)
    bright_img = cv.imread('pout-bright.jpg', 0)

    # hashmaps to store the pixel counts (intensity count)
    dark_dict = {i:0 for i in range(256)}
    bright_dict = {i:0 for i in range(256)}

    # normalisation of the intensity values to get the Probability Distribution Function (pdf)
    for i in dark_img.flatten():
        dark_dict[i] = dark_dict[i] + 1
    for i in bright_img.flatten():
        bright_dict[i] = bright_dict[i] + 1
    for i in range(256):
        dark_dict[i] = dark_dict[i] / dark_img.size
    for i in range(256):
        bright_dict[i] = bright_dict[i] / bright_img.size

    #finding the histogram equal pdf and histogram specific pdf
    equalised_img, pdf_dark = histogram_equalisation(dark_dict, dark_img)
    specific_img, match_table = histogram_matching(dark_dict, bright_dict, dark_img, bright_img)

    # plot of all images
    plt.figure()
    plt.subplot(141);plt.imshow(dark_img, cmap = "grey"); plt.title("Dark Image")
    plt.subplot(142); plt.imshow(bright_img, cmap = "grey"); plt.title("Bright Image")
    plt.subplot(143); plt.imshow(equalised_img, cmap = "grey"); plt.title("Equalised Image")
    plt.subplot(144); plt.imshow(specific_img, cmap = "grey"); plt.title("Specific Image")

    # plot of pdf of all images
    plt.figure()
    plt.subplot(141); plt.bar(dark_dict.keys(), dark_dict.values(), color = "skyblue");plt.xlabel('Values[0-255]');plt.ylabel('Intensity count');plt.title('Dark Image');plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(142); plt.bar(bright_dict.keys(), bright_dict.values(), color = "magenta");plt.xlabel('Values[0-255]');plt.ylabel('Intensity count');plt.title('Bright Image');plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(143); plt.bar(pdf_dark.keys(), pdf_dark.values(), color = "teal");plt.xlabel('Values[0-255]');plt.ylabel('Intensity count');plt.title('Equalised Image');plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(144); plt.bar(match_table.keys(), match_table.values(), color = "teal");plt.xlabel('Values[0-255]');plt.ylabel('Intensity count');plt.title('Specific Image');plt.grid(axis='y', linestyle='--', alpha=0.7)

    e, s = builtin_processing(dark_img, bright_img, equalised_img, specific_img)
    print(f"The accuracy of User defined code,\nEqualisation Transform :{100 - e:.2f}%\nSpecification Transform :{100 - s}%")

    plt.show()
