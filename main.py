import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


def convert2GrayScale(image_path):
    image = cv.imread(image_path)

    if image is None:
        print("Error: Unable to read the image.")
        return

    height, width, channels = image.shape

    grayscale_image = image.copy()

    for y in range(height):
        for x in range(width):
            #RGB values of the pixel
            r, g, b = image[y, x]

            #average method gray_value = int((r + g + b) / 3)

            #luminosity formula
            gray_value = int(r * 299/1000 + g * 587/1000 + b * 114/1000)

            #set grayscale value for pixel in the new image                                   
            grayscale_image[y, x] = gray_value

    cv.imwrite("photos/gray_image.jpg", grayscale_image)
    return "photos/gray_image.jpg"

def convert2BinaryImage(image_path, threshold=128):
    grayscale_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if grayscale_image is None:
        print("Error: Unable to read the image.")
        return None

    binary_image = grayscale_image.copy()

    # Apply thresholding
    binary_image[binary_image < threshold] = 0
    binary_image[binary_image >= threshold] = 255

    cv.imwrite("photos/binary_image.jpg", binary_image)
    return "photos/binary_image.jpg"

def resizeImage(image_path, target_height, target_width):
    image=cv.imread(image_path)
    height, width, channels = image.shape
    
    # Initialize the resized image
    resized_image = np.zeros((target_height, target_width, channels), dtype=np.uint8)
    
    # Compute scaling factors
    scale_x = width / target_width
    scale_y = height / target_height
    
    # Iterate over each pixel in the resized image
    for y in range(target_height):
        for x in range(target_width):
            # Calculate the corresponding pixel position in the original image
            original_x = int(x * scale_x)
            original_y = int(y * scale_y)
            
            # Copy the value of the nearest neighbor pixel from the original image
            resized_image[y, x] = image[original_y, original_x]
    
    cv.imwrite("photos/resize_Image.jpg", resized_image)
    return "photos/resize_Image.jpg"  

def calculate_histogram(gray_image, num_bins=256):
    # Initialize histogram with zeros
    histogram = np.zeros(num_bins, dtype=int)
    
    # Iterate through each pixel in the image
    for row in gray_image:
        for pixel in row:
            # Increment the corresponding bin in the histogram
            histogram[pixel] += 1
            
    return histogram

def calculate_metrics(gray_image):

    mean_intensity = np.mean(gray_image)
    std_deviation = np.std(gray_image)

    # Calculate histogram
    histogram=calculate_histogram(gray_image)

    # Calculate normalized histogram
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    normalized_histogram = histogram / total_pixels

    # Calculate cumulative histogram
    cumulative_histogram = np.cumsum(histogram)

    # Calculate entropy
    entropy = -np.sum(normalized_histogram * np.log2(normalized_histogram + 1e-10))

    return mean_intensity, std_deviation, entropy, histogram, normalized_histogram, cumulative_histogram

def show_metrics(mean_intensity, std_deviation, entropy, histogram, normalized_histogram, cumulative_histogram):
    print(f"Mean intensity: {mean_intensity}")
    print(f"Standard Deviation (STD): {std_deviation}")
    print(f"Entropy: {entropy}")

    # Plot histogram
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    # plt.plot(histogram)
    plt.bar(np.arange(256), histogram, color='gray')
    plt.title("Histogram")
    plt.xlabel("Intensity level")
    plt.ylabel("Frequency")

    # Plot normalized histogram
    plt.subplot(1, 3, 2)
    # plt.plot(normalized_histogram)
    plt.bar(np.arange(256), normalized_histogram, color='gray')
    plt.title("Normalized Histogram")
    plt.xlabel("Intensity level")
    plt.ylabel("Probability")

    # Plot cumulative histogram
    plt.subplot(1, 3, 3)
    # plt.plot(cumulative_histogram, color='black')
    plt.bar(np.arange(256), cumulative_histogram, color='black')
    plt.title("Cumulative Histogram")
    plt.xlabel("Intensity level")
    plt.ylabel("Cumulative Frequency")

    # plt.tight_layout()
    plt.show()

def calculate_and_show_metrics(image_path):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Unable to read the image.")
        return

    # Calculate metrics
    mean_intensity, std_deviation, entropy, histogram, normalized_histogram, cumulative_histogram = calculate_metrics(gray_image)

    # Display metrics
    show_metrics(mean_intensity, std_deviation, entropy, histogram, normalized_histogram, cumulative_histogram)

def enhance_contrast(image_path):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Unable to read the image.")
        return None

    # Apply histogram equalization
    equalized_image = cv.equalizeHist(gray_image)

    enhanced_image_path = "photos/C.jpg"
    cv.imwrite(enhanced_image_path, equalized_image)

    return enhanced_image_path

def flip_image(image_path):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Unable to read the image.")
        return None

    flipped_image = np.zeros_like(gray_image)

    # Flip the image horizontally manually
    height, width = gray_image.shape
    for y in range(height):
        for x in range(width):
            flipped_image[y, width - 1 - x] = gray_image[y, x]

    flipped_image_path = "photos/flipped.jpg"
    cv.imwrite(flipped_image_path, flipped_image)

    return flipped_image_path

def blur_image(image_path):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Unable to read the image.")
        return None

    # Define a blur kernel
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9  # Normalization to maintain intensity values

    # Convolve the image with the kernel manually
    blurred_image = cv.filter2D(gray_image, -1, kernel)

    blurred_image_path = "photos/blured.jpg"
    cv.imwrite(blurred_image_path, blurred_image)

    return blurred_image_path

def create_negative_image(image_path):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Unable to read the image.")
        return None

    # Invert the intensity values to create the negative image
    negative_image = 255 - gray_image

    negative_image_path = "photos/negative.jpg"
    cv.imwrite(negative_image_path, negative_image)

    return negative_image_path

def crop_image(image_path, x, y, w, h):
    image = cv.imread(image_path)

    if image is None:
        print("Error: Unable to read the image.")
        return None

    # Ensure that the crop region is within the bounds of the image
    max_y, max_x, _ = image.shape
    if x < 0 or y < 0 or x + w > max_x or y + h > max_y:
        print("Error: Crop region exceeds image boundaries.")
        return None

    # Crop the image
    cropped_image = image[y:y+h, x:x+w]

    cropped_image_path = "photos/cropped_image.jpg"
    cv.imwrite(cropped_image_path, cropped_image)

    return cropped_image_path

def compute_histogram(image):
    # Compute the histogram of the grayscale image
    hist = cv.calcHist([image], [0], None, [256], [0, 256])

    return hist

def find_best_matching_strip(image_path, cropped_strip_path):
    cropped_strip = cv.imread(cropped_strip_path, cv.IMREAD_GRAYSCALE)

    if cropped_strip is None:
        print("Error: Unable to read the cropped strip image.")
        return None

    # Compute the histogram of the cropped strip
    hist_cropped_strip = compute_histogram(cropped_strip)

    # Read the original grayscale image
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Unable to read the original grayscale image.")
        return None

    # Initialize variables to store best matching strip and its similarity score
    best_matching_strip = None
    best_matching_score = -1  # Initialize with a low value

    # Iterate through each vertical strip in image and compare histograms
    max_y, max_x = gray_image.shape  # Get only height and width
    for x in range(max_x - cropped_strip.shape[1] + 1):  # Adjust the range for x
        # Extract the current vertical strip
        current_strip = gray_image[:, x:x+cropped_strip.shape[1]]

        # Compute histogram of the current strip
        hist_current_strip = compute_histogram(current_strip)

        # Compute histogram similarity (using intersection for simplicity)
        similarity = cv.compareHist(hist_cropped_strip, hist_current_strip, cv.HISTCMP_INTERSECT)

        # Update best matching strip and score if similarity is higher
        if similarity > best_matching_score:
            best_matching_score = similarity
            best_matching_strip = current_strip

    # Write best matching strip to a file
    best_matching_strip_path = "photos/best_matching_strip_image.jpg"
    cv.imwrite(best_matching_strip_path, best_matching_strip)

    return best_matching_strip_path


imagePath = "photos/lena.jpg"

grayScaleImagePath=convert2GrayScale(imagePath)
grayScaleImage=cv.imread(grayScaleImagePath)
cv.imshow("Grayscale Image", grayScaleImage)

binaryImagePath=convert2BinaryImage(grayScaleImagePath)
binaryImage=cv.imread(binaryImagePath)
cv.imshow("Binary Image", binaryImage)

resizeImagePath=resizeImage(grayScaleImagePath,256,256)
resizeimage=cv.imread(resizeImagePath)
cv.imshow("re", resizeimage)


enhanced_image_path = enhance_contrast(grayScaleImagePath)
enhanced_image = cv.imread(enhanced_image_path)
cv.imshow("Enhanced Image", enhanced_image)

flipped_image_path = flip_image(grayScaleImagePath)
flipped_image = cv.imread(flipped_image_path)
cv.imshow("Flipped Image", flipped_image)

blurred_image_path = blur_image(grayScaleImagePath)
blurred_image = cv.imread(blurred_image_path)
cv.imshow("Blurred Image", blurred_image)

negative_image_path = create_negative_image(grayScaleImagePath)
negative_image = cv.imread(negative_image_path)
cv.imshow("Negative Image", negative_image)

x = 100  # Starting horizontal position
y = 50   # Starting vertical position
w = 200  # Width of the cropped image
h = 150  # Height of the cropped image

cropped_image_path = crop_image(grayScaleImagePath, x, y, w, h)
cropped_image = cv.imread(cropped_image_path)
cv.imshow("Cropped Image", cropped_image)

cropped_strip_path = crop_image(grayScaleImagePath, x, y, w, h)
best_matching_strip = find_best_matching_strip(grayScaleImagePath, cropped_strip_path)
best_matching_strip = cv.imread(best_matching_strip)
cv.imshow("Best Matching Strip", best_matching_strip)

calculate_and_show_metrics(grayScaleImagePath)

cv.waitKey(0)
cv.destroyAllWindows()


