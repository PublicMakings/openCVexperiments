""" This code shows a way of getting the digit's edges in a pre-defined position 
(in green)
    from https://stackoverflow.com/questions/51363676/how-to-edit-image-with-opencv-to-read-text-with-ocr?r=SearchResults
    """
import cv2
import numpy as np


def find_template(src_gray, src_template_gray, n_matches_min):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    """ find grid using SIFT """
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(src_template_gray, None)
    kp2, des2 = sift.detectAndCompute(src_gray, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > n_matches_min:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h_template, w_template = src_template_gray.shape
        pts = np.float32([[0, 0], [0, h_template - 1], [w_template - 1, h_template - 1], [w_template - 1,0]]).reshape(-1,1,2)
        homography = cv2.perspectiveTransform(pts, M)
    else:
        print "Not enough matches are found - %d/%d" % (len(good), n_matches_min)
        matchesMask = None

    # show matches
    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                     singlePointColor = None,
                     matchesMask = matchesMask, # draw only inliers
                     flags = 2)

    if matchesMask:
        src_gray_copy = src_gray.copy()
        sift_matches = cv2.polylines(src_gray_copy, [np.int32(homography)], True, 255, 2, cv2.LINE_AA)
    sift_matches = cv2.drawMatches(src_template_gray, kp1, src_gray_copy, kp2, good, None, **draw_params)
    return sift_matches, homography


def transform_perspective_and_crop(homography, src, src_gray, src_template_gray):
    """ get mask and contour of template """
    mask_img_template = np.zeros(src_gray.shape, dtype=np.uint8)
    mask_img_template = cv2.polylines(mask_img_template, [np.int32(homography)], True, 255, 1, cv2.LINE_AA)
    _ret, contours, hierarchy = cv2.findContours(mask_img_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = None
    # approximate the contour
    c = contours[0]
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our template
    warp = None
    if len(approx) == 4:
        template_contour = approx
        cv2.drawContours(mask_img_template, [template_contour] , -1, (255,0,0), -1)
        """ Transform perspective """
        # now that we have our template contour, we need to determine
        # the top-left, top-right, bottom-right, and bottom-left
        # points so that we can later warp the image -- we'll start
        # by reshaping our contour to be our finals and initializing
        # our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = template_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        homography = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, homography)
        warp = cv2.warpPerspective(src, M, (maxWidth, maxHeight))

        # resize warp
        h_template, w_template, _n_channels = src_template_gray.shape
        warp = cv2.resize(warp, (w_template, h_template), interpolation=cv2.INTER_AREA)

    return warp


def crop_img_in_hsv_range(img, hsv, lower_bound, upper_bound):
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    # do an MORPH_OPEN (erosion followed by dilation) to remove isolated pixels
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    return mask, res


def separate_rois(column_mask, img_gray):
    # go through each of the boxes
    # https://stackoverflow.com/questions/41592039/contouring-a-binary-mask-with-opencv-python
    border = cv2.copyMakeBorder(column_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    _, contours, hierarchy = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(-1, -1))
    cell_list = []
    for contour in contours:
        cell_mask = np.zeros_like(img_gray) # Create mask where white is what we want, black otherwise
        cv2.drawContours(cell_mask, [contour], -1, 255, -1) # Draw filled contour in mask

        # turn that mask into a rectangle
        (x,y,w,h) = cv2.boundingRect(contour)
        #print("x:{} y:{} w:{} h:{}".format(x, y, w, h))
        cv2.rectangle(cell_mask, (x, y), (x+w, y+h), 255, -1)
        # copy the img_gray using that mask
        img_tmp_region = cv2.bitwise_and(img_gray, img_gray, mask= cell_mask)

        # Now crop
        (y, x) = np.where(cell_mask == 255)
        (top_y, top_x) = (np.min(y), np.min(x))
        (bottom_y, bottom_x) = (np.max(y), np.max(x))
        img_tmp_region = img_tmp_region[top_y:bottom_y+1, top_x:bottom_x+1]

        cell_list.append([img_tmp_region, top_x, top_y])

    return cell_list


""" 1. Load images """
# load image of plate
src_path = "nRHzD.jpg"
src = cv2.imread(src_path)

# load template of plate (to be looked for)
src_template_path = "nRHzD_template.jpg"
src_template = cv2.imread(src_template_path)

""" 2. Find the plate (using the template image) and crop it into a rectangle """
# convert images to gray scale
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_template_gray = cv2.cvtColor(src_template, cv2.COLOR_BGR2GRAY)
# use SIFT to find template
n_matches_min = 10
template_found, homography = find_template(src_gray, src_template_gray, n_matches_min)
warp = transform_perspective_and_crop(homography, src, src_gray, src_template)

warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
warp_hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
template_hsv = cv2.cvtColor(src_template, cv2.COLOR_BGR2HSV)

""" 3. Find regions of interest (using the green parts of the template image) """
green_hsv_lower_bound = [50, 250, 250]
green_hsv_upper_bound = [60, 255, 255]
mask_rois, mask_rois_img = crop_img_in_hsv_range(warp, template_hsv, green_hsv_lower_bound, green_hsv_upper_bound)
roi_list = separate_rois(mask_rois, warp_gray)
# sort the rois by distance to top right corner -> x (value[1]) + y (value[2])
roi_list = sorted(roi_list, key=lambda values: values[1]+values[2])

""" 4. Apply a Canny Edge detection to the rois (regions of interest) """
for i, roi in enumerate(roi_list):
    roi_img, roi_x_offset, roi_y_offset = roi
    print("#roi:{} x:{} y:{}".format(i, roi_x_offset, roi_y_offset))

    roi_img_blur_threshold = cv2.Canny(roi_img, 40, 200)
    cv2.imshow("ROI image", roi_img_blur_threshold)
    cv2.waitKey()
