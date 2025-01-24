import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import cv2 as cv
import sys, math, copy

lena_org_img = cv.imread('Lena.png')
b, g, r = cv.split(lena_org_img)

n = len(r)

lena_pgm_img = np.zeros(r.shape)
lena_pgm_img = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

lena_gray_img = cv.cvtColor(lena_org_img, cv.COLOR_BGR2GRAY)
lena_org_img = cv.cvtColor(lena_org_img, cv.COLOR_BGR2RGB)

# plt.subplot(151);plt.imshow(r); plt.title("Red")
# plt.subplot(152);plt.imshow(g); plt.title("Green")
# plt.subplot(153);plt.imshow(b); plt.title("Blue")
plt.figure(figsize = (18, 3))
plt.subplot(161);plt.imshow(lena_org_img);plt.title("Original RGB image"); plt.axis('off')
plt.subplot(162);plt.imshow(lena_pgm_img,cmap = "gray");plt.title("Calculated Gray Scale Image"); plt.axis('off')
# plt.subplot(143);plt.imshow(lena_gray_img,cmap = "gray");plt.title("Actual Gray Scale Image")

# print(lena_gray_img-lena_pgm_img)
diff = (lena_gray_img-lena_pgm_img)/lena_gray_img
# print(len(diff[0]))
error = np.sum(diff[0]) / len(diff[0])
# print(np.sum(diff[0]), error)
print(f"The accuracy between actual and calculated gray image (pixels) : {(100-(error* 100))} %")
print('''\nThe slight loss in the accuracy is due to the usage of plt.imshow() method that expects the data type of the pixel to be in 8-bit integer. 

Since we performed certain calculations the resultant pixels that were in 32-bit float has been down casted to 8 bit values that caused loss in 1 pixel at places.''')

# plt.subplots_adjust(hspace=1.0, wspace=0.4)  
# print(lena_org_img)

lena_shp = list(lena_org_img.shape)
# print(type(lena_shp))

lena_shp[0] = 2 * lena_shp[0]
lena_shp[1] = 2 * lena_shp[1]

# Nearest Neighbour Computation
def nninterpolation():
    lena_nn_image = np.zeros(tuple(lena_shp))
    for i in range(len(lena_org_img)):
        for j in range(len(lena_org_img)):
            lena_nn_image[2*i][2*j] = lena_org_img[i][j]

            for l in range(2 * i, (2 * i)+2):
                for k in range(2 * j, (2 * j)+2):
                    lena_nn_image[l][k] = lena_nn_image[2 * i][2 * j]

    return lena_nn_image
lena_nn_image = nninterpolation()
lena_nn_image = np.astype(lena_nn_image, np.uint8)

plt.subplot(163);plt.imshow(lena_nn_image);plt.title("Nearest Neighbour"); plt.axis('off')

# Bi linear interpolation
def bilinear():
    lena_bilinear_img = np.zeros(tuple(lena_shp))

    original_height, original_width = lena_org_img.shape[0], lena_org_img.shape[1]

    for i in range(len(lena_bilinear_img)):
        for j in range(len(lena_bilinear_img)):
            new_i = i/2
            new_j = j/2

            x1 = math.floor(new_i)
            y1 = math.floor(new_j)

            x2 = min(x1+1, original_height-1)
            y2 = min(y1+1, original_width-1)

            a = new_i - x1
            b = new_j - y1

            lena_bilinear_img[i][j] = (a * b * lena_org_img[x2, y2] + (1-a) * b * lena_org_img[x2, y1] + a * (1-b) * lena_org_img[x1, y2] + (1-a) * (1-b) * lena_org_img[x1, y1])

    return lena_bilinear_img.astype(np.uint8)
lena_bilinear_img = bilinear()
plt.subplot(164);plt.imshow(lena_bilinear_img);plt.title("Bilinear Image"); plt.axis('off')

def bilinear_builtin():
    builtin_size = tuple([lena_org_img.shape[1], lena_org_img.shape[0]])
    lena_bilinear_builtin = cv.resize(lena_org_img, builtin_size, cv.INTER_LINEAR)
    return lena_bilinear_builtin

lena_bilinear_builtin = bilinear_builtin()
plt.subplot(165);plt.imshow(lena_bilinear_builtin); plt.title("Bilinear Built in"); plt.axis('off')

def bicubic():
    builtin_size = tuple([lena_org_img.shape[1], lena_org_img.shape[0]])
    lena_bicubic_builtin = cv.resize(lena_org_img, builtin_size, cv.INTER_CUBIC)
    return lena_bicubic_builtin
lena_bicubic_builtin = bicubic()
plt.subplot(166);plt.imshow(lena_bicubic_builtin); plt.title("Bicubic Built in"); plt.axis('off')

plt.tight_layout()
plt.show()