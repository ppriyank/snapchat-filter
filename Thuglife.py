import  time
import  datetime as  dt

import  cv2
import  dlib
import  numpy

import  sys

PREDICTOR_PATH  =  "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR  =  1  
FEATHER_AMOUNT  =  11

FACE_POINTS = list(range(17,  68))
MOUTH_POINTS  =  list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17,  22))
LEFT_BROW_POINTS  =  list(range(22, 27))
RIGHT_EYE_POINTS  =  list(range(36, 42))
LEFT_EYE_POINTS = list(range(42,  48))
NOSE_POINTS = list(range(27,  35))
JAW_POINTS  =  list(range(0,  17))

ALIGN_POINTS  =  (list(range(0, 68))  )
OVERLAY_POINTS  =  [
      LEFT_EYE_POINTS + RIGHT_EYE_POINTS  +  LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
]

COLOUR_CORRECT_BLUR_FRAC  =  0.6

detector  =  dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
      rects = detector(im,  0) 
      if  len(rects) !=  1  :  
            return  None    
      return  numpy.matrix([[p.x,  p.y] for p in  predictor(im,  rects[0]).parts()])

def get_image_mask(im):
  #  mask = cv2.Mat(im.size(),  cv2.CV_8UC3)  # TODO: Can change  to 1 Channel
  mask = numpy.zeros((im.shape[0], im.shape[1],  3),  dtype=numpy.uint8)
  for  i  in range(im.shape[0]):
    for j in  range(im.shape[1]):
      mask_value = im[i, j,  3]/255
      mask[i,  j] = [mask_value,  mask_value,  mask_value]
  return mask
      
def transformation_from_points(points1, points2):
      points1 = points1.astype(numpy.float64)
      points2 = points2.astype(numpy.float64)

      c1  =  numpy.mean(points1,  axis=0)
      c2  =  numpy.mean(points2,  axis=0)
      points1 -=  c1
      points2 -=  c2

      s1  =  numpy.std(points1)
      s2  =  numpy.std(points2)
      points1 /=  s1
      points2 /=  s2

      U,  S, Vt  =  numpy.linalg.svd(points1.T * points2)
      R = (U  *  Vt).T

      return  numpy.vstack([numpy.hstack(((s2  /  s1)  *  R,
                                                           c2.T - (s2 / s1) * R * c1.T)),
                                      numpy.matrix([0.,  0.,  1.])])

def warp_im(im, M,  dshape):
      output_im = numpy.zeros(dshape, dtype=im.dtype)
      cv2.warpAffine(im,
                             M[:2],
                             (dshape[1],  dshape[0]),
                             dst=output_im,
                             borderMode=cv2.BORDER_TRANSPARENT,
                             flags=cv2.WARP_INVERSE_MAP)
      return  output_im

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video-1488820746.mp4')
frame = None  
# temp_im = cv2.imread("teeth main.png",  cv2.IMREAD_UNCHANGED)
temp_im2= cv2.imread("thug_life1.png", cv2.IMREAD_UNCHANGED)
mask1 = get_image_mask(temp_im2)   # TODO: Check the speed
temp_im3= cv2.imread("thug_life2.png",  cv2.IMREAD_UNCHANGED)
mask2 = get_image_mask(temp_im3)   # TODO: Check the speed
temp_im4= cv2.imread("thug_life3.png",  cv2.IMREAD_UNCHANGED)
mask3 = get_image_mask(temp_im4)   # TODO: Check the speed

# im2 = cv2.imread("teeth main.png",  cv2.IMREAD_COLOR)
im3 = cv2.imread("thug_life1.png", cv2.IMREAD_COLOR)
im4 = cv2.imread("thug_life2.png",  cv2.IMREAD_COLOR)
im5 = cv2.imread("thug_life3.png",  cv2.IMREAD_COLOR)
# im2 = cv2.resize(im2, (720, 360))

landmarks2  =  numpy.load("face_main.npy")
landmarks2  =  numpy.asmatrix(landmarks2)

# prev_time = time.time()
im2 = im3
mask = mask1
i = 0 

while(cap.isOpened()):  
    if cv2.waitKey(1)  &  0xFF ==  ord('q'):
        break
    ret, im1 = cap.read()
    # if ret ==  False  :  
    #   break
    landmarks1 = get_landmarks(im1)
    if landmarks1  != None  :
        M = transformation_from_points(landmarks1[ALIGN_POINTS],  landmarks2[ALIGN_POINTS])
        if i == 0 : 
          im2 = im3
          i = 1
          mask = mask1
        elif i == 1: 
          i = 2 
          mask = mask2
          im2 = im4
        else : 
          i= 0
          mask = mask3
          im2 = im5

        warped_mask = warp_im(mask, M,  im1.shape)
        warped_im2  =  warp_im(im2, M,  im1.shape)  
        im1 = (im1  *  (1.0 - warped_mask)  +  warped_im2 * warped_mask)/255
        cv2.imshow('frame2',im1)
          
    # curr_time  =  time.time()
    # time_diff  =  curr_time  -  prev_time
    # print  (1 / time_diff), format(cap.get(cv2.CAP_PROP_FPS))
    # prev_time  =  curr_time
