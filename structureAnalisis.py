import csv
import math

import cv2 as cv
import numpy as np


def structure_analsis():
    # Do 10 kolorow!
    pixels = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (125, 125, 255), (125, 255, 125), (255, 125, 125)]

    global_contours = []
    global_contours_labeling = []
    # global_hierarchy = []

    # image pick
    image = cv.imread(r'genmesh.png')

    # Create csv
    f = open(r'data.csv', 'w', encoding="UTF-8", newline='')
    headers = ["ID", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10"]
    writer = csv.writer(f)
    writer.writerow(headers)

    # Larger image for labels
    labeled_image = cv.resize(image, (600, 600), interpolation=cv.INTER_NEAREST_EXACT)

    # mask for 9 colors (increase to 10)
    for i in range(1, 10):
        # Color mask
        mask = cv.inRange(image, pixels[i], pixels[i])
        mask_labeled = cv.inRange(labeled_image, pixels[i], pixels[i])
        # Apply a mask
        masked_img = cv.bitwise_and(image, image, mask=mask)
        masked_labeled_img = cv.bitwise_and(labeled_image, labeled_image, mask=mask_labeled)
        # binarization but first convert to grayscale
        greyscale = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)
        greyscale_labeled = cv.cvtColor(masked_labeled_img, cv.COLOR_BGR2GRAY)
        # binarization of image
        flag, binned = cv.threshold(greyscale, 10, 255, cv.THRESH_BINARY)
        flag, binned_labeled = cv.threshold(greyscale_labeled, 10, 255, cv.THRESH_BINARY)
        # contours binding
        color_contours, trash = cv.findContours(binned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_for_labeling, tmp = cv.findContours(binned_labeled, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Save contours
        global_contours.append(color_contours)
        global_contours_labeling.append(contours_for_labeling)


    i = 0
    for contour in global_contours_labeling:
        for c in contour:
            # get minimal rect
            rect = cv.minAreaRect(c)
            # get center
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            # labels for grain
            cv.putText(labeled_image, text=str(i), org=(cx - 7, cy + 5), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.3, color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)
            # increment the counter

            i += 1

    # reset i
    i = 0

    for contour in global_contours:
        for c in contour:
            # get dimensions
            rect = cv.minAreaRect(c)
            w = rect[1][1]
            h = rect[1][0]

            # grain params
            ID = i
            # temp of image
            temp_image = np.zeros_like(image)
            # draw contour mask
            cv.drawContours(temp_image, [c], 0, (255, 255, 255), -1)
            # get list opf points
            points = np.where(temp_image == 255)
            # combine x and y
            contour_insides = list(zip(points[1], points[0]))
            # remove duplicate
            contour_insides = list(dict.fromkeys(contour_insides))

            # W1
            area = cv.contourArea(c)
            W1 = 2 * np.sqrt(area / np.pi)

            # W2
            perimeter = cv.arcLength(c, True)
            W2 = perimeter / np.pi

            # W3
            W3 = (perimeter / (2 * np.sqrt(np.pi * area))) - 1

            # W4
            moments = cv.moments(c)
            # calculate center
            x_center = int(moments["m10"] / moments["m00"])
            y_center = int(moments["m01"] / moments["m00"])
            # save center
            center = [x_center, y_center]
            # sum of center
            sum_dist = 0.0
            for pixel in contour_insides:
                dist = math.dist(pixel, center)
                sum_dist += (dist * dist)
            W4 = area / (np.sqrt(2 * np.pi * sum_dist))

            # W5
            min_dist = 0.0
            for inner_pixels in contour_insides:
                distances = []
                for contours_pixels in c:
                    distances.append(math.dist(contours_pixels[0], inner_pixels))
                min_dist += np.amin(distances)
            W5 = (np.power(area, 3) / np.power(min_dist, 2))

            # W6
            sum_dis = 0.0
            sum_dis_sqrt = 0.0
            distances = []
            for pixel in c:
                # from center
                dist = math.dist(center, pixel[0])
                # save
                distances.append(dist)
                # sum all
                sum_dis += dist
                sum_dis_sqrt += np.power(dist, 2)
            W6 = np.sqrt((sum_dis * sum_dis) / (len(c) * sum_dis_sqrt - 1))

            # W7
            r_min = np.amin(distances)
            r_max = np.amax(distances)
            W7 = r_min / r_max

            # W8
            maxD = h if h > w else w
            W8 = maxD / perimeter

            # W9
            W9 = (2 * np.sqrt(np.pi * area)) / perimeter

            # W10
            W10 = h / w

            # add to csv
            data_w = [ID, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10]
            writer.writerow(data_w)

            i += 1


    cv.imwrite(r"labeled.png", labeled_image)

    f.close()
