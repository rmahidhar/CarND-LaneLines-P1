import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
#from IPython.display import HTML

import math
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # print(lines)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    return lines


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def imshow(image, title, plot_num):
    plt.subplot(plot_num)
    plt.title(title)
    plt.imshow(image)


def imshow_gray(image, title, plot_num):
    plt.subplot(plot_num)
    plt.title(title)
    plt.imshow(image, cmap='gray')


class FindLaneLine(object):
    GAUSSIAN_KERNEL_SIZE = 5
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi/180
    HOUGH_THRESHOLD = 25
    HOUGH_MIN_LINE_LEN = 30
    HOUGH_MAX_LINE_GAP = 200
    MAX_SLOPE = 0.8
    MIN_SLOPE = 0.4

    def __init__(self, image):
        # self.image = mpimg.imread(image)
        self.image = image
        self.processed_image = np.copy(self.image)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.ratio = self.height / self.width + 0.050

    def left_line_slope_filter(m):
        if (m < 0):
            if math.fabs(m) >= FindLaneLine.MIN_SLOPE and math.fabs(m) <= FindLaneLine.MAX_SLOPE:
                return True
        return False

    def right_line_slope_filter(m):
        if (m > 0):
            if m >= FindLaneLine.MIN_SLOPE and m <= FindLaneLine.MAX_SLOPE:
                return True
        return False

    def slope(line):
        # print(line)
        if ((line[3] - line[1]) == 0 or ((line[2] - line[0]) == 0)):
            # print(line, None)
            return None
        size = math.hypot(line[2] - line[0], line[3] - line[1])
        # print(line, (line[3] - line[1])/(line[2] - line[0]), size)
        return (line[3] - line[1]) / (line[2] - line[0])

    def filter_hough_lines(lines, filter_slope):
        filtered_lines = []
        for line in lines:
            line = line[0]
            m = FindLaneLine.slope(line)
            if (m and filter_slope(m)):
                filtered_lines.append(line)
        return (filtered_lines if len(filtered_lines) else None)

    def apply_linalg(lines):
        """
            linear regression least squares alogrithm
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
        """
        np_lines = np.asarray(lines)
        x = np.reshape(np_lines[:, [0, 2]], (1, len(lines) * 2))[0]
        y = np.reshape(np_lines[:, [1, 3]], (1, len(lines) * 2))[0]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        x = np.array(x)
        y = np.array(x * m + c)
        return x, y, m, c

    def region_of_interest_left(self):
        left_bottom = [self.width / 10, self.height]
        left_top = [(1 - self.ratio) * self.width, self.ratio * self.height]
        right_top = [self.width / 2, self.ratio * self.height]
        right_bottom = [self.width / 2, self.height]
        roi = np.array([left_bottom, left_top, right_top, right_bottom], np.int32)
        return region_of_interest(self.processed_image, [roi])

    def region_of_interest_right(self):
        left_bottom = [self.width / 2, self.height]
        left_top = [self.width / 2, self.ratio * self.height]
        right_bottom = [self.width, self.height]
        right_top = [self.ratio * self.width, self.ratio * self.height]
        roi = np.array([left_bottom, left_top, right_top, right_bottom], np.int32)
        return region_of_interest(self.processed_image, [roi])

    def pre_process_image(self):
        plt.subplots(nrows=3, ncols=3)
        plt.tight_layout()

        imshow(self.image, 'Original Image', 331)

        self.processed_image = grayscale(self.image)
        imshow_gray(self.processed_image, 'GrayScale Image', 332)

        self.processed_image = gaussian_blur(self.processed_image, FindLaneLine.GAUSSIAN_KERNEL_SIZE)
        imshow_gray(self.processed_image, 'Gaussian Blur Image', 333)

    def draw_lines(self, lines, color=[255, 0, 0], thickness=2):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if lines:
            for line in lines:
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
        return img

    def draw_lines_applying_linalg(self):
        left_line_x, left_line_y, left_line_m, left_line_c = FindLaneLine.apply_linalg(self.lines_left)
        right_line_x, right_line_y, right_line_m, right_line_c = FindLaneLine.apply_linalg(self.lines_right)

        """
            min y value in both left and right lines gives the top y co-ordinate value
            lower the top y by 50 pixels for smaller line.
        """
        top_y = np.min([np.min(left_line_y), np.min(right_line_y)])
        # max y value in both left and right lines gives the bottom y co-ordinate value
        bottom_y = np.max([np.max(left_line_y), np.max(right_line_y)])

        """
            right line top co-oridnate x value is derived from top_y, right line slope and intercept
            using line equation, y = mx + c, x = (y-c)/m
        """
        top_right = np.array([(top_y - right_line_c) / right_line_m, top_y], dtype=int)
        """
            left line top co-oridnate x value is derived from top_y, left line slope and intercept
            using line equation, y = mx + c, x = (y-c)/m
        """
        top_left = np.array([(top_y - left_line_c) / left_line_m, top_y], dtype=int)

        """
            left line bootom co-oridnate x value is derived from bottom_y, left line slope and intercept
            using line equation, y = mx + c, x = (y-c)/m
        """
        bottom_left = np.array([(bottom_y - left_line_c) / left_line_m, bottom_y], dtype=int)
        """
            right line bootom co-oridnate x value is derived from bottom_y, right line slope and intercept
            using line equation, y = mx + c, x = (y-c)/m
        """
        bottom_right = np.array([(bottom_y - right_line_c) / right_line_m, bottom_y], dtype=int)

        line_img = np.zeros(self.image.shape, dtype=np.uint8)
        # draw left line from bottom_left to top_left points
        line_img = cv2.line(line_img, (bottom_left[0], bottom_left[1]), (top_left[0], top_left[1]), [255, 0, 0], 10)
        # draw right line from bottom_left to top_left points
        line_img = cv2.line(line_img, (bottom_right[0], bottom_right[1]), (top_right[0], top_right[1]), [255, 0, 0], 10)
        out = weighted_img(line_img, self.image)
        return out

    def process_image(self):
        self.pre_process_image()

        self.processed_image = canny(self.processed_image,
                                     FindLaneLine.CANNY_LOW_THRESHOLD,
                                     FindLaneLine.CANNY_HIGH_THRESHOLD)
        imshow_gray(self.processed_image, 'Canny Image', 334)

        # get the left line region of interest
        left_roi = self.region_of_interest_left()
        imshow_gray(left_roi, 'Left ROI', 335)

        # get the right line region of interest
        right_roi = self.region_of_interest_right()
        imshow_gray(right_roi, 'Right ROI', 336)

        # apply hough transfrom on the left line region
        hough_left = hough_lines(left_roi,
                                 FindLaneLine.HOUGH_RHO,
                                 FindLaneLine.HOUGH_THETA,
                                 FindLaneLine.HOUGH_THRESHOLD,
                                 FindLaneLine.HOUGH_MIN_LINE_LEN,
                                 FindLaneLine.HOUGH_MAX_LINE_GAP)
        # print("hough_lines_left")
        # print(hough_left)
        # filter out the lines with positive slopes and out of range slopes on the left line region
        self.lines_left = FindLaneLine.filter_hough_lines(hough_left,
                                                          lambda m: FindLaneLine.left_line_slope_filter(m))
        line_img = self.draw_lines(self.lines_left)
        imshow_gray(line_img, 'Hough Left ROI', 337)

        # apply hough transfrom on the right line region
        hough_right = hough_lines(right_roi,
                                  FindLaneLine.HOUGH_RHO,
                                  FindLaneLine.HOUGH_THETA,
                                  FindLaneLine.HOUGH_THRESHOLD,
                                  FindLaneLine.HOUGH_MIN_LINE_LEN,
                                  FindLaneLine.HOUGH_MAX_LINE_GAP)
        # print("hough_lines_right")
        # print(hough_right)

        # filter out the lines with negitive slopes and out of range slopes on the left line region
        self.lines_right = FindLaneLine.filter_hough_lines(hough_right,
                                                           lambda m: FindLaneLine.right_line_slope_filter(m))
        line_img = self.draw_lines(self.lines_right)
        imshow_gray(line_img, 'Hough Right ROI', 338)

        """
            draw lines applying linear regression least squares algorithm
            on left and right lines.
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
        """
        if self.lines_left is not None and self.lines_right is not None:
            image = self.draw_lines_applying_linalg()
            imshow(image, 'Result', 339)
            return image
        else:
            imshow(self.image, 'Result', 339)
            return self.image


def process_test_images():
    output_dir = "test_images_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image in os.listdir("test_images/"):
        print("Processing image ", image)
        img = mpimg.imread('test_images/' + image)
        name = image.split('.')[0]
        plt.figure()
        plt.title('Input:' + name)
        plt.imshow(img)

        lanefind = FindLaneLine(img)
        result = lanefind.process_image()

        fig = plt.figure()
        plt.title('Result: ' + name + ' detected lanes')
        plt.imshow(result)
        mpimg.imsave(output_dir + '/' + image, result)

def process_image(image):
    lanefind = FindLaneLine(image)
    result = lanefind.process_image()
    return result

def process_video_file(input, output):
    output_file = output
    clip1 = VideoFileClip(input)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_file, audio=False)

def main():
    process_test_images()
    process_video_file("solidWhiteRight.mp4", "white.mp4")
    process_video_file("solidYellowLeft.mp4", "yellow.mp4")
    process_video_file("challenge.mp4", "extra.mp4")

if __name__ == "__main__":
    main()