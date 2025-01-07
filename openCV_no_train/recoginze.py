import openCV_no_train
import os
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 60)




fps = video_capture.get(cv2.CAP_PROP_FPS)
print(f"Actual Camera FPS: {fps}")


# Load sample pictures and learn to recognize them
image_directory = r"D:\face_recognize\recognize_git\recognition_image_list"

# Initialize the known face encodings and names arrays
known_face_encodings = []
known_face_names = []

# Traverse the directory and process all .jpg images
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        # Build the full file path
        image_path = os.path.join(image_directory, filename)

        # Load the image and get face encoding
        image = openCV_no_train.load_image_file(image_path)
        face_encodings = openCV_no_train.face_encodings(image)

        # Ensure there is at least one face encoding found
        if face_encodings:
            known_face_encodings.append(face_encodings[0])  # Take the first face encoding
            known_face_names.append(
                os.path.splitext(filename)[0])  # Remove the file extension and use the filename as name

# Initialize some variables

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
similarity = 0
# 初始化标志，只处理每隔一帧
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if process_this_frame:
        # 进行人脸识别处理
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 缩小图像处理

        # Convert the image from BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]

        # 查找人脸位置和编码
        face_locations = openCV_no_train.face_locations(rgb_small_frame)
        face_encodings = openCV_no_train.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = openCV_no_train.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = openCV_no_train.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                similarity = (1 - face_distances[best_match_index]) * 100
                similarity = round(similarity, 2)
                print(f"Similarity for {name}: {similarity}%")

            face_names.append(name)

    process_this_frame = not process_this_frame  # 切换处理标志

    # 显示图像并继续处理
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name}: {similarity}%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
