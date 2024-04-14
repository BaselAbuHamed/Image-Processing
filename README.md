# Image Processing Project

This project is aimed at performing various image processing tasks using OpenCV and NumPy libraries in Python. It includes functions for converting images to grayscale, binary, and resizing, as well as performing operations such as histogram calculation, contrast enhancement, flipping, blurring, creating negative images, cropping, and finding the best matching strip in an image.

***

# Features

- **Convert to Grayscale:** Convert images to grayscale using the luminosity formula.
- **Convert to Binary:** Convert grayscale images to binary using thresholding.
- **Resize Images:** Resize images to specified dimensions using nearest-neighbor interpolation.
- **Calculate Metrics:** Compute metrics such as mean intensity, standard deviation, entropy, and histograms.
- **Enhance Contrast:** Enhance image contrast using histogram equalization.
- **Flip Images:** Flip images horizontally.
- **Blur Images:** Apply a blur filter to images.
- **Create Negative Images:** Generate negative images by inverting intensity values.
- **Crop Images:** Crop images to specified regions.
- **Find Best Matching Strip:** Find the best matching strip in an image based on histogram similarity.

***

# Usage

To use the project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/image-processing-project.git

2. Install the required dependencies:

    ```bash
    pip install opencv-python numpy matplotlib

3. Run the main.py script to execute image processing tasks:

    ```bash
    python main.py

***

# Example Images

1. **Original Image:**
   
This is the original input image used for image processing tasks.

![lena](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/97f92bfa-15f4-4cfa-b659-47ec0884080b)


2. **Greyscale Image**:

This image shows the result of converting the original image to grayscale.

![gray_image](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/d24dff8d-829f-452f-910b-a35f8d07076f)

3. **Binary Image:**

This image displays the result of converting the grayscale image to binary using thresholding.

![binary_image](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/c9fe3016-1d87-4bd1-bcb6-d0f64239487f)


4. **Resized Image:**

 This image shows the result of resizing the original image to the specified dimensions.

![resize_Image](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/7244c41e-7f9a-41bb-9cda-3b4011a70c08)

5. **Enhanced Image:**

This image displays the result of enhancing the contrast of the grayscale image using histogram equalization.

![C](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/298972dc-2dc0-41b9-b525-354b503d6994)

6. **Flipped Image:**

This image shows the result of flipping the grayscale image horizontally.

![flipped](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/5e446607-c77e-4df2-b241-50956feedbac)

7. **Blurred Image:**

This image displays the result of applying a blur filter to the grayscale image.

![blured](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/7ad0cb4e-766c-4d0a-aa59-607bf4463247)


8. **Negative Image:**

This image shows the result of generating a negative image by inverting the intensity values of the grayscale image.

![negative](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/56d02826-6df6-481a-86cb-91a004dfa846)

9. **Cropped Image:**

This image displays a cropped region of the original image.

![cropped_image](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/fd883a78-432e-4bb4-a145-e727c2e81204)

10. **Best Matching Strip:**

This image shows the best matching strip found in the original image based on histogram similarity.

![best_matching_strip_image](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/42e1eb4b-4e79-4e1f-968a-a2625221b381)

***

# Histogram, Normalized Histogram, and Cumulative Histogram

In image processing, a histogram is a graphical representation of the distribution of pixel intensity values in an image. It provides valuable insights into the tonal distribution of an image, which can be useful for various analysis and enhancement tasks. The histogram typically plots the frequency of occurrence of each intensity level (ranging from 0 to 255 for 8-bit images) along the horizontal axis against the corresponding number of pixels having that intensity level along the vertical axis.

## Histogram Calculation

To compute a histogram for an image, we iterate through each pixel and increment the corresponding bin in the histogram array. For grayscale images, the histogram will have 256 bins, each representing an intensity level from 0 to 255.

## Normalized Histogram

The normalized histogram is obtained by dividing the frequency of each intensity level by the total number of pixels in the image. This normalization ensures that the sum of all bin values in the histogram equals 1, representing the probability distribution of pixel intensities.

## Cumulative Histogram

The cumulative histogram (also known as the cumulative distribution function) is derived from the normalized histogram. It represents the cumulative sum of frequencies up to each intensity level. The cumulative histogram provides information about the cumulative distribution of pixel intensities in an image, which can be useful for tasks such as contrast stretching and histogram equalization.

![hist+norm+cumu](https://github.com/BaselAbuHamed/Image-Processing/assets/107325485/e8ed5313-8ce6-4d54-8934-0505b2f16443)

By understanding histograms, normalized histograms, and cumulative histograms, image processing practitioners can gain valuable insights into the characteristics of images and perform a wide range of enhancement and analysis tasks effectively.
