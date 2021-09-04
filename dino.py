#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import hypot
import pyautogui
import dlib


# In[3]:


# Required file path
model_weights =  "dependencies/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# architecture file path
model_arch = "dependencies/deploy.prototxt.txt"

# Load the caffe model
net = cv2.dnn.readNetFromCaffe(model_arch, model_weights)


# In[4]:


def face_detector(image, threshold =0.7):
    
    # Get the height,width of the image
    h, w = image.shape[:2]

    # Apply mean subtraction, and create 4D blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0,(300, 300), (104.0, 117.0, 123.0))
    
    # Set the new input value for the network
    net.setInput(blob)
    
    # Run forward pass on the input to compute output
    faces = net.forward()
    
    # Get the confidence value for all detected faces
    prediction_scores = faces[:,:,:,2]
    
    # Get the index of the prediction with highest confidence 
    i = np.argmax(prediction_scores)
    
    # Get the face with highest confidence 
    face = faces[0,0,i]
    
    # Extract the confidence
    confidence = face[2]
    
    # if confidence value is greater than the threshold
    if confidence > threshold:
        
        # The 4 values at indexes 3 to 6 are the top-left, bottom-right coordinates
        # scales to range 0-1.The original coordinates can be found by 
        # multiplying x,y values with the width,height of the image
        box = face[3:7] * np.array([w, h, w, h])
        
        # The coordinates are the pixel numbers relative to the top left
        # corner of the image therefore needs be quantized to int type
        (x1, y1, x2, y2) = box.astype("int")
        
        # Draw a bounding box around the face.
        annotated_frame = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)
        output = (annotated_frame, (x1, y1, x2, y2), True, confidence)
    
    # Return the original frame if no face is detected with high confidence.
    else:
        output = (image,(),False, 0)
    
    
    return output


# In[5]:


# Get the video feed from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal one so we can adjust it
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL) 

while(True):
    
    # Read the frames
    ret, frame = cap.read()
    
    # Break if frame is not returned
    if not ret:
        break
        
    # Flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Detect face in the frame
    annotated_frame, coords, status, conf = face_detector(frame)
    
    # Display the frame
    cv2.imshow('Face Detection', annotated_frame)
    
    # Break the loop if ESC key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()


# In[6]:


# initialize the landmark detector

predictor = dlib.shape_predictor("dependencies/shape_predictor_68_face_landmarks.dat")


# In[7]:


def detect_landmarks(box, image):
    
    # For faster results convert the image to gray-scale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the coordinates 
    (x1, y1, x2, y2) = box

    # Perform the detection
    shape = predictor(gray_scale, dlib.rectangle(x1, y1, x2, y2))
    
    # Get the numPy array containing the coordinates of the landmarks
    landmarks = shape_to_np(shape)
    
   # Draw the landmark points with circles 
    for (x, y) in landmarks:
        annotated_image = cv2.circle(image, (x, y),2, (0, 127, 255), -1)

    return annotated_image, landmarks


# In[8]:


def shape_to_np(shape): 
    
    # Create an array of shape (68, 2) for storing the landmark coordinates
    landmarks = np.zeros((68, 2), dtype="int")
    
    # Write the x,y coordinates of each landmark into the array
    for i in range(0, 68): 
        landmarks[i] = (shape.part(i).x, shape.part(i).y)
        
        
    return landmarks


# In[9]:


# Get the video feed from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal one so we can adjust it
cv2.namedWindow('Landmark Detection', cv2.WINDOW_NORMAL) 

while(True):
    # Read the frames
    ret, frame = cap.read()
    
    # Break if frame is not returned
    if not ret:
        break
        
    # Flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Detect face in the frame
    face_image, box_coords, status, conf = face_detector(frame)
    
    if status:
        
        # Get the landmarks for the face region in the frame
        landmark_image, landmarks = detect_landmarks(box_coords, frame)

    # Display the frame
    cv2.imshow('Landmark Detection',landmark_image)
    
    # Break the loop if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()


# 
# 
# 
# 

# In[10]:


def is_mouth_open(landmarks, ar_threshold = 0.7): 
    
    
    # Calculate the euclidean distance labelled as A,B,C
    A = hypot(landmarks[50][0] - landmarks[58][0], landmarks[50][1] - landmarks[58][1])
    B = hypot(landmarks[52][0] - landmarks[56][0], landmarks[52][1] - landmarks[56][1])
    C = hypot(landmarks[48][0] - landmarks[54][0], landmarks[48][1] - landmarks[54][1])
    
    # Calculate the mouth aspect ratio
    # The value of vertical distance A,B is averaged
    mouth_aspect_ratio = (A + B) / (2.0 * C)
    
    # Return True if the value is greater than the threshold
    if mouth_aspect_ratio > ar_threshold:
        return True, mouth_aspect_ratio
    else:
        return False, mouth_aspect_ratio


# In[11]:


# Get the video feed from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal one so we can adjust it
cv2.namedWindow('Mouth Status', cv2.WINDOW_NORMAL)

while(True):
    # Read the frames
    ret, frame = cap.read()
    
    # Break if frame is not returned
    if not ret:
        break
        
    # Flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Detect face in the frame
    face_image, box_coords, status, conf = face_detector(frame)
    
    if status:
        
        # Get the landmarks for the face region in the frame
        landmark_image, landmarks = detect_landmarks(box_coords, frame)
        
        # Adjust the threshold and make sure it's working for you.
        mouth_status,_ = is_mouth_open(landmarks, ar_threshold = 0.6)
        
        # Display the mouth status
        cv2.putText(frame,'Is Mouth Open: {}'.format(mouth_status),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255),2)


    # Display the frame
    cv2.imshow('Mouth Status',frame)
    
    # Break the loop if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()


# In[12]:


def face_proximity(box,image, proximity_threshold = 325):
    
    # Get the height and width of the face bounding box
    face_width =  box[2]-box[0]
    face_height = box[3]-box[1]
    
    # Draw rectangle to guide the user 
    # Calculate the angle of diagonal using face width, height 
    theta = np.arctan(face_height/face_width)
     
    # Use the angle to calculate height, width of the guide rectangle
    guide_height = np.sin(theta)*proximity_threshold
    guide_width  = np.cos(theta)*proximity_threshold
    
    # Calculate the mid-point of the guide rectangle/face bounding box
    mid_x,mid_y = (box[2]+box[0])/2 , (box[3]+box[1])/2
    
    #Calculate to coordinates of top-left and bottom-right corners
    guide_topleft = int(mid_x-(guide_width/2)), int(mid_y-(guide_height/2))
    guide_bottomright = int(mid_x +(guide_width/2)), int(mid_y + (guide_height/2))
    
    # Draw the guide rectangle
    cv2.rectangle(image, guide_topleft, guide_bottomright, (0, 255, 255), 2)
    
    # Calculate the diagonal distance of the bounding box
    diagonal = hypot(face_width, face_height)
    
    # Return True if distance greater than the threshold
    if diagonal > proximity_threshold:
        return True, diagonal
    else:
        return False, diagonal


# In[13]:


# Get the video feed from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal one so we can adjust it
cv2.namedWindow('Face proximity', cv2.WINDOW_NORMAL)

while(True):
    
    # Read the frames
    ret, frame = cap.read()
    
    # Break if frame is not returned
    if not ret:
        break
        
    # Flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Detect face in the frame
    face_image, box_coords, status, conf = face_detector(frame)
    
    if status:
        
        # Check if face is closer than the defined threshold
        is_face_close,_ = face_proximity(box_coords, face_image, proximity_threshold = 325)
        
        # Display the mouth status
        cv2.putText(face_image,'Is Face Close: {}'.format(is_face_close),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255),2)

        
    # Display the frame
    cv2.imshow('Face proximity',face_image)
    
    # Break the loop if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()


# In[14]:


# Get the video feed from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal one so we can adjust it
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)

while(True):
    
    # Read the frames
    ret, frame = cap.read()
    
    # Break if frame is not returned
    if not ret:
        break
        
    # Flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Detect face in the frame
    face_image, box_coords, status, conf = face_detector(frame)
    
    if status:
        
        # Detect landmarks if the frame is found
        landmark_image, landmarks = detect_landmarks(box_coords, frame)
        
        # Get the current mouth aspect ratio
        _,mouth_ar = is_mouth_open(landmarks)
    
        # Get the current face proximity
        _, proximity  = face_proximity(box_coords, face_image)

        # Calculate threshold values
        ar_threshold = mouth_ar*1.4
        proximity_threshold = proximity*1.3

        
        # Dsiplay the threshold values
        cv2.putText(frame, 'Aspect ratio threshold: {:.2f} '.format(ar_threshold), 
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255),2)
        
        cv2.putText(frame,'Proximity threshold: {:.2f}'.format(proximity_threshold), 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255),2)
     
    # Display the frame
    cv2.imshow('Calibration',frame)
    
    # Break the loop if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()


# In[15]:


# This will open a context menu
pyautogui.click(button='right')


# In[16]:


# Press space bar. This will scroll down the page in some browsers
pyautogui.press('space')


# In[21]:


# This will move the focus to the next cell in the notebook
pyautogui.press(['shift','enter'])


# In[22]:


# Hold down the shift key
pyautogui.keyDown('shift')


# In[23]:


# Press enter while the shift key is down, this will run the next code cell
pyautogui.press('enter')

# Release the shift key
pyautogui.keyUp('shift')


# In[24]:


# This will run automatically after running the two code cells above
print('I ran')


# In[27]:


# Get the video feed from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal one so we can adjust it
cv2.namedWindow('Lets Go! Play the Game', cv2.WINDOW_NORMAL)

# By default each key press is followed by a 0.1 second pause
pyautogui.PAUSE = 0.0

# The fail-safe triggers an exception in case mouse
# is moved to corner of the screen
#pyautogui.FAILSAFE = False

while(True):
    
     # Read the frames
    ret, frame = cap.read()
    
    # Break if frame is not returned
    if not ret:
        break
        
    # Flip the frame
    frame = cv2.flip( frame, 1 )
    
    # Detect face in the frame
    face_image, box_coords, status, conf = face_detector(frame)
    
    if status:
        
        # Detect landmarks if a face is found
        landmark_image, landmarks = detect_landmarks(box_coords, frame)
        
        # Check if mouth is open
        is_open,_ = is_mouth_open(landmarks, ar_threshold)
        
        # If the mouth is open trigger space key Down event to jump
        if is_open:
            
            pyautogui.keyDown('space')
            mouth_status = 'Open'

        else:
            # Else the space key is Up
            pyautogui.keyUp('space')
            mouth_status = 'Closed'
        
        # Display the mouth status on the frame
        cv2.putText(frame,'Mouth: {}'.format(mouth_status),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255),2)
        
        # Check the proximity of the face
        is_closer,_  = face_proximity(box_coords, frame, proximity_threshold)
        
        # If face is closer press the down key
        if is_closer:
            pyautogui.keyDown('down')
            
        else:
            pyautogui.keyUp('down')
        
    # Display the frame
    cv2.imshow('Dino with OpenCV',frame)

    # Break the loop if ESC key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

