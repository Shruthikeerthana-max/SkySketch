import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize deques and indices for - blue, green, red, purple colors
gpoints = [deque(maxlen=1024)]
pipoints = [deque(maxlen=1024)]
ppoints = [deque(maxlen=1024)]
opoints = [deque(maxlen=1024)]

# Indices for deque
green_index = 0
pink_index = 0
purple_index = 0
orange_index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

colors = [(0, 255, 0), (255, 105, 180), (128, 0, 128), (255, 165, 0)]  # Green, Pink, Purple, Orange
colorIndex = 0

paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (255, 105, 180), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (128, 0, 128), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (255, 165, 0), 2)

# Filled buttons
cv2.rectangle(paintWindow, (40, 1), (140, 65), (200, 200, 200), -1)
cv2.rectangle(paintWindow, (160, 1), (255, 65), (0, 255, 0), -1)
cv2.rectangle(paintWindow, (275, 1), (370, 65), (255, 105, 180), -1)
cv2.rectangle(paintWindow, (390, 1), (485, 65), (128, 0, 128), -1)
cv2.rectangle(paintWindow, (505, 1), (600, 65), (255, 165, 0), -1)

# Button Text
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "LAVENDER", (285, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "PURPLE", (415, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (510, 33), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 2, cv2.LINE_AA)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# initialize mediapipe, specify number of hands and the sensitivity
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
      # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Drawing buttons on frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (200, 200, 200), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (255, 105, 180), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (128, 0, 128), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (255, 165, 0), 2)

    # Button Text
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "LAVENDER", (285, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "PURPLE", (415, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (510, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

     # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

# Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        print(center[1] - thumb[1])
        if (thumb[1] - center[1] < 30):
            gpoints.append(deque(maxlen=512))
            green_index += 1
            pipoints.append(deque(maxlen=512))
            pink_index += 1
            ppoints.append(deque(maxlen=512))
            purple_index += 1
            opoints.append(deque(maxlen=512))
            orange_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                gpoints = [deque(maxlen=512)]
                pipoints = [deque(maxlen=512)]
                ppoints = [deque(maxlen=512)]
                opoints = [deque(maxlen=512)]

                green_index = 0
                pink_index = 0
                purple_index = 0
                orange_index = 0

                paintWindow[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # green
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # pink
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # purple
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # orange
        else:
            if colorIndex == 0:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 1:
                pipoints[pink_index].appendleft(center)
            elif colorIndex == 2:
                ppoints[purple_index].appendleft(center)
            elif colorIndex == 3:
                opoints[orange_index].appendleft(center)
    # Append the next deques when nothing is detected to avoid messing up
    else:
        gpoints.append(deque(maxlen=512))
        green_index += 1
        pipoints.append(deque(maxlen=512))
        pink_index += 1
        ppoints.append(deque(maxlen=512))
        purple_index += 1
        opoints.append(deque(maxlen=512))
        orange_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [gpoints, pipoints, ppoints, opoints]
   
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
