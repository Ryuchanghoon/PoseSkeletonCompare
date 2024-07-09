import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7)


image_path = 'test_data/frame_1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


results = pose.process(image_rgb)

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    
    x = [landmark.x for landmark in landmarks]
    y = [landmark.y for landmark in landmarks]

    x = np.array(x)
    y = np.array(y)

    
    plt.figure()
    plt.scatter(x, y, c='r', marker='o')

    
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        plt.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], 'b-')

    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # y축 반전. 이미지와 동일하게.

    plt.show()

else:
    print("insane image path")
