import sys
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')  # Or use 'Agg' for non-interactive use
import matplotlib.pyplot as plt

print("All images shall be rendered  in 640 x 800 resolution")

n = len(sys.argv)

if(n == 1):
    print("Provide the image path as commandline argument. Exiting!!!")
else:
    print(f"The location of the image is : {sys.argv[1]}")
    imgpath = sys.argv[1]

    # print(type(imgpath))
    print("How do you want to render the input image :\n(0 - GrayScale\n1 - Color without transparency)\n")
    render = int(input("Value: "))

    img = cv.imread(imgpath, render)

    if(render == 1):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #fixing image size for all inputs
    imgsize = (640, 800)

    resizedimg = cv.resize(img, dsize=imgsize, interpolation=cv.INTER_AREA)

    # print(type(resizedimg[0][0]))

    # blurredimg = cv.blur(resizedimg, (200, 200))
    print("\n")
    x_kernel = int(input("Enter the x value for kernel (only odd values e.g, 51) : "))
    y_kernel = int(input("Enter the y value for kernel (only odd values) e.g, 51: "))

    gaussianblur = cv.GaussianBlur(resizedimg, (x_kernel, y_kernel), 0)

    mean = int(input("\nEnter the mean for sampling Gaussian noise from Normal Distribution : "))
    std_dev = int(input("Enter the standard deviation : "))

    gaussiannoise = np.random.normal(mean, std_dev, gaussianblur.shape) 
    noisyimg = gaussianblur + gaussiannoise
    noisyimg = np.clip(noisyimg, 0, 255).astype(np.uint8)

    # medianblur = cv.medianBlur(resizedimg, 201)
    # bilateralblur = cv.bilateralFilter(resizedimg, 15, 201, 201)

    plt.subplot(151);plt.imshow(img, cmap = "gray"); plt.title("Actual Image")
    plt.subplot(152);plt.imshow(resizedimg, cmap = "gray"); plt.title("Resized Image")
    plt.subplot(153);plt.imshow(gaussianblur, cmap = "gray");plt.title("Blurred Image")
    plt.subplot(154); plt.imshow(noisyimg, cmap = "gray"); plt.title("Noisy Image")

    #image compression jpeg 2000 then decompress to 70-80%

    compressedimg = "compressed.jp2"
    cv.imwrite(compressedimg, noisyimg, [cv.IMWRITE_JPEG2000_COMPRESSION_X1000, 70]) #70% compression quality

    #decompression
    decompressedimg = cv.imread(compressedimg, render)

    # print(resizedimg.dtype)
    plt.subplot(155); plt.imshow(decompressedimg, cmap = "gray");plt.title("Decompressed Image")

    # PEAK SIGNAL TO NOISE RATIO CALCULATION
    print(
    '''
    \n
    PSNR > 40 DISTORTED IMAGE IS ALMOST SAME AS ORIGINAL\n
    PSNR < 40 AND > 30 SLIGHT DISTORTION\n
    PSNR > 20 AND < 30 NOTICEABLE DISTORTION\n
    PSNR < 20 QUALITY IS POOR  \n
    '''
    )

    L = 256  # maximum value considering grayscale rendering of input images
    MSE = np.mean((resizedimg - decompressedimg) ** 2)
    peak_signal_to_noise_ratio = 10 * np.log10(((L-1)**2) / MSE)

    print(f"The calculated PSNR : {peak_signal_to_noise_ratio} dB")

    plt.show()