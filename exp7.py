# Create the mask for the glasses
imgGlassesGray = cv2.cvtColor(imgGlasses, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(imgGlassesGray, 10, 255, cv2.THRESH_BINARY)
#orig_mask = imgGlasses[:,:,3]

# Create the inverted mask for the glasses
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert glasses image to BGR
# and save the original image size (used later when re-sizing the image)
imgGlasses = imgGlasses[:,:,0:3]
origGlassesHeight, origGlassesWidth = imgGlasses.shape[:2]

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

        for (ex, ey, ew, eh) in eyes:
            glassesWidth = 2*ew
            glassesHeight = glassesWidth * origGlassesHeight / origGlassesWidth

            # Center the glasses
            x1 = ex - (glassesWidth/4)
            x2 = ex + ew + (glassesWidth/4)
            y1 = ey + eh - (glassesHeight/2)
            y2 = ey + eh + (glassesHeight/2)

            # Re-calculate the width and height of the glasses image
            glassesWidth = x2 - x1
            glassesHeight = y2 - y1

            # Re-size the original image and the masks to the glasses sizes
            # calcualted above
            glasses = cv2.resize(imgGlasses, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)

            # take ROI for glasses from background equal to size of glasses image
            roi = roi_color[y1:y2, x1:x2]

                        # roi_bg contains the original image only where the glasses is not
            # in the region that is the size of the glasses.
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            # roi_fg contains the image of the glasses only where the glasses is
            roi_fg = cv2.bitwise_and(glasses,glasses,mask = mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst

            break
