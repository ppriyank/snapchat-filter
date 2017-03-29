# TODO: Just for benchmarking remove at the end
import time
import datetime as dt

import cv2
import dlib
import numpy

import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
# ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
#                                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
ALIGN_POINTS = (list(range(0, 2)) + list(range(15, 27))  + list(range(38, 48)) + [30, 32 ] )
# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# class TooManyFaces(Exception):
#     pass

# class NoFaces(Exception):
#     pass

def get_landmarks(im):
    rects = detector(im, 0)
    
    if len(rects) != 1 : 
        return None   
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def get_image_mask(im):
	# mask = cv2.Mat(im.size(), cv2.CV_8UC3)	# TODO: Can change to 1 Channel
	mask = numpy.zeros((im.shape[0], im.shape[1], 3), dtype=numpy.uint8)
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			mask_value = im[i, j, 3]/255
			mask[i, j] = [mask_value, mask_value, mask_value]
	return mask
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0) )
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    # np.add(a, b, out=a, casting="unsafe")
    
    temp = 128 * (im2_blur <= 1.0)
    im2_blur = im2_blur + temp
    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))
cap = cv2.VideoCapture(0)
frame = None 
# im = cv2.imread("Sunglasses-PNG-File.png", cv2.IMREAD_COLOR)
temp_im = cv2.imread("Sunglasses-PNG-File.png", cv2.IMREAD_UNCHANGED)
temp_im2 = cv2.resize(temp_im, (720, 360))
mask = get_image_mask(temp_im2)	# TODO: Check the speed

im = cv2.imread("Sunglasses-PNG-File.png", cv2.IMREAD_COLOR)
im2 = cv2.resize(im, (720, 360))

landmarks2 = numpy.load("sunglasses.npy")
landmarks2 = numpy.asmatrix(landmarks2)

# TODO: Benchmarking
prev_time = time.time()

while(cap.isOpened()):	# TODO: Remove constant assignments from the loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	ret, im1 = cap.read()
	
#	cv2.imshow('frame',frame)
	
	output_im = im1	# TODO: Redundant remove it
	landmarks1 = get_landmarks(im1)
	
	if landmarks1 != None :   
	
		M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
		
		# mask = get_face_mask(im2, landmarks2)	# TODO: Check the speed
		# cv2.imshow('frame3', mask)
		warped_mask = warp_im(mask, M, im1.shape)
		# combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
		
		warped_im2 = warp_im(im2, M, im1.shape)
		# cv2.imshow('frame5', im2)
		# cv2.imshow('frame4', warped_im2)
		# warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
		
		output_im = (im1 * (1.0 - warped_mask) + warped_im2 * warped_mask)/255
	
		# cv2.imwrite('output.jpg', output_im)
		# cv2.waitKey(1)
	# break
	cv2.imshow('frame2',output_im)

	# TODO: Benchmarking
	curr_time = time.time()
	time_diff = curr_time - prev_time
	print (1 / time_diff), format(cap.get(cv2.CAP_PROP_FPS))
	prev_time = curr_time


