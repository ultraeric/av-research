import cv2
import numpy as np
from numpy import linalg
import imageio
import time
import math


class _PixelLine:
    def __init__(self, x_0, y_0, x_1, y_1, diag_length):
        self.p_0, self.p_1 = ((x_0, y_0), (x_1, y_1)) if x_0 < x_1 else ((x_1, y_1), (x_0, y_0))
        self.p_0, self.p_1 = np.array(self.p_0), np.array(self.p_1)
        self.slope = float(y_1 - y_0) / float(x_1 - x_0 + 1e-32)
        self.theta = math.atan(self.slope)
        self.diag_length = diag_length

    def perp_dist(self, x, y):
        point = np.array([x, y])
        return linalg.norm(np.cross(self.p_1 - self.p_0, self.p_0 - point)) / linalg.norm(self.p_1 - self.p_0)

    def similar_angle(self, pixel_line, merge_leniency=5./360.):
        return abs(self.theta - pixel_line.theta) <= merge_leniency * 2 * np.pi

    def perp_near(self, pixel_line, merge_leniency=5./360.):
        perp_dist = max(min(self.perp_dist(*pixel_line.p_0),
                            self.perp_dist(*pixel_line.p_1)),
                        min(pixel_line.perp_dist(*self.p_0),
                            pixel_line.perp_dist(*self.p_1)))
        return perp_dist < self.diag_length * merge_leniency

    def can_combine(self, pixel_line, merge_leniency=5./360.):
        return self.similar_angle(pixel_line, merge_leniency) and self.perp_near(pixel_line, merge_leniency)

    def get_combinable(self, pixel_lines, merge_leniency=5./360.):
        return set([pixel_line for pixel_line in pixel_lines if
                    (self.can_combine(pixel_line, merge_leniency) and self != pixel_line)])


class LaneDetector:
    def __init__(self, gaussian_kernel=(5, 5), canny_thresholds=(40, 120), pixel_acc=2,
                 angle_acc=np.pi/45, vote_leniency=0.1, merge_leniency=1./36.):
        self.gaussian_kernel = gaussian_kernel
        self.canny_thresholds = canny_thresholds
        self.pixel_acc = pixel_acc
        self.angle_acc = angle_acc
        self.vote_leniency = vote_leniency
        self.merge_leniency = merge_leniency

    def detect(self, img, roi_only=False):
        """
        Detects the lane the vehicle is currently in, if any.

        :param img: <np.array> image in a numpy array. Should be <height x width x depth>
        :return: <np.array>, <tuple>
        a new image with the detected lanes in red, as well as the coordinates for the center line of the lane.
        """
        img = np.copy(img)

        # Mask lane colors, gaussian blur,
        img_color_masked = self.apply_lane_color_mask(img)
        img_edges = self.apply_canny([img_color_masked])
        img_roi = self.apply_roi_mask(img_edges)

        # Hough transform
        vote_leniency = round(self.vote_leniency * self.get_img_diag(img))
        lines = cv2.HoughLinesP(img_roi,
                                rho=self.pixel_acc,
                                theta=self.angle_acc,
                                threshold=vote_leniency,
                                minLineLength=vote_leniency / 8.,
                                maxLineGap=vote_leniency / 2.)
        lines = [] if lines is None else lines
        lines = [line[0] for line in lines]

        for line in lines:
            x_0, y_0, x_1, y_1 = line
            cv2.line(img, (x_0, y_0), (x_1, y_1), (0, 255, 0), 3)

        if roi_only:
            return img

        aggregate_lines = self.aggregate_lines(img, lines)
        lines = self.polyfit_lines(img, aggregate_lines)


        for line in lines:
            x_0, y_0, x_1, y_1 = line
            cv2.line(img, (x_0, y_0), (x_1, y_1), (255, 0, 0), 2)

        return img

    def apply_canny(self, imgs=()):
        if type(imgs) is not tuple and type(imgs) is not list:
            imgs = [imgs]
        net_img = np.zeros(imgs[0].shape[:2], dtype=np.uint8)
        for img in imgs:
            img = cv2.GaussianBlur(img, self.gaussian_kernel, 0)
            img = cv2.Canny(img,
                            self.canny_thresholds[0],
                            self.canny_thresholds[1],
                            apertureSize=3)
            net_img = cv2.bitwise_or(net_img, img)
        return net_img

    def apply_local_norm(self, img):
        float_img = img.astype(np.float) / 255.
        float_blurred = cv2.GaussianBlur(float_img, (0, 0), sigmaX=2, sigmaY=2)
        float_numerator = float_img - float_blurred
        float_blurred = cv2.GaussianBlur(float_numerator * float_numerator, (0, 0), sigmaX=20, sigmaY=20)
        float_denominator = cv2.pow(float_blurred, 0.5)
        float_img = float_numerator / float_denominator
        return (cv2.normalize(float_img, dst=float_img, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX) * 255).astype(np.uint8)

    def apply_roi_mask(self, img, roi_mask=None):
        """
        Cuts out the region of interest and returns the resulting image

        :param img: <np.array>
        :return: <np.array>
        """
        roi_mask = roi_mask or self.get_roi_mask(img)
        img_roi = cv2.bitwise_and(img, roi_mask)

        return img_roi

    def apply_lane_color_mask(self, img, lane_color_mask=None):
        """
        Applies the lane color mask and returns the resulting image

        :param img: <np.array>
        :return: <np.array>
        """
        lane_color_mask = lane_color_mask or self.get_lane_color_mask(img)
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_masked = cv2.bitwise_and(grey_img, lane_color_mask)

        return img_masked

    def get_roi_mask(self, img):
        """
        Get region of interest mask

        :param img: <np.array> Image to mask
        :return: <np.array>
        """
        shape = img.shape
        mask = np.zeros(shape, np.uint8)
        shape = [np.array([
                           (0, round(shape[0] * 0.75)),
                           (0, shape[0]),
                           (shape[1], shape[0]),
                           (shape[1], round(shape[0] * 0.75)),
                           (round(shape[1] * 0.6), round(shape[0] * 0.666)),
                           (round(shape[1] * 0.4), round(shape[0] * 0.666))
                        ])]
        cv2.fillPoly(mask, shape, 255)
        return mask

    def get_lane_color_mask(self, img):
        """
        Calculates the lane color mask and returns it

        :param img: <np.array>
        :return: <np.array>
        """

        # Generate two derivative images, an HSV and a Greyscale
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Set the lower and upper bound yellow colors, then get the masks for the yellow and white colors
        yellow_lower, yellow_upper = np.array([20, 100, 100], np.uint8), np.array([30, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(grey_img, 150, 255)

        # Calculate the mask (either yellow or white)
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        return lane_mask

    def get_img_diag(self, img):
        shape = img.shape
        return (shape[0] ** 2 + shape[1] ** 2) ** 0.5

    def aggregate_lines(self, img, lines):
        """
        Aggregates the lines and returns the whole lane lines

        :param img: <np.array>
        :param lines: <list of list of [x_0, y_0, x_1, y_1] list>
        :return:
        """
        if len(lines) == 0:
            return []
        lines = [_PixelLine(*line, self.get_img_diag(img)) for line in lines]
        lines = [line for line in lines if line.theta >= np.pi / 9. or line.theta <= -np.pi / 9.]
        unique_lines = set(lines)
        merge_leniency = self.merge_leniency
        line_sets = []
        while len(unique_lines) > 0:
            curr_root = unique_lines.pop()
            curr_set = {curr_root}
            fringe = curr_root.get_combinable(unique_lines, merge_leniency)
            unique_lines = unique_lines.difference(fringe)
            while len(fringe) > 0:
                curr_root = fringe.pop()
                fringe_update = curr_root.get_combinable(unique_lines, merge_leniency)
                fringe = fringe.union(fringe_update)
                unique_lines = unique_lines.difference(fringe_update)
                curr_set.add(curr_root)
            line_sets.append(curr_set)
        return line_sets

    def polyfit_lines(self, img, aggregate_lines):
        final_lines = []
        height, width, depth = img.shape
        y_upper, y_lower = (height * 2) // 3, round(height)
        for line_set in aggregate_lines:
            x = []
            y = []
            for line in line_set:
                x.extend([line.p_0[0], line.p_1[0]])
                y.extend([line.p_0[1], line.p_1[1]])
            line = np.polyfit(x, y, 1)
            a, b = line[0], line[1]
            x_0, y_0 = min(max(((y_upper - b) / a).astype(np.int), 1), width), y_upper
            x_1, y_1 =  min(max(((y_lower - b) / a).astype(np.int), 1), width), y_lower
            final_lines.append((x_0, y_0, x_1, y_1))
        return final_lines


def unit_test():
    lane_detector = LaneDetector()
    img = imageio.imread('LOCATION TO PICTURE')

    result_img = lane_detector.detect(img)
    imageio.imwrite('LOCATION TO PICTURE', result_img)

    img = imageio.imread('LOCATION TO PICTURE')
    t = time.time()
    lane_detector.detect(img)
    t = time.time() - t

    print('Lane Detection Max Theoretical FPS: ', 1./t)


unit_test()
