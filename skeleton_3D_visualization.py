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
    z = [landmark.z for landmark in landmarks]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], [z[start_idx], z[end_idx]], 'b-')

    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

else:
    print("insane image path")
