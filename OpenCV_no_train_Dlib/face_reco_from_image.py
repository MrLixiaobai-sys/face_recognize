import dlib
import numpy as np
import pandas as pd
import os
import logging
from PIL import Image
from mtcnn import MTCNN


# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
# detector = dlib.get_frontal_face_detector()
detector = MTCNN()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class FaceRecognizer:
    def __init__(self):
        self.face_feature_known_list = []  # 用来存放所有录入人脸特征的数组 / Save the features of faces in database
        self.face_name_known_list = []  # 存储录入人脸名字 / Save the name of faces in database

    # 从 "features_all.csv" 读取录入人脸特征 / Read known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append(0.0)
                    else:
                        features_someone_arr.append(float(csv_rd.iloc[i][j]))
                self.face_feature_known_list.append(features_someone_arr)
            logging.info("Faces in Database：%d", len(self.face_feature_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            return 0

    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 从图片中识别人脸并输出到控制台
    def recognize_faces_from_image(self, image_path):
        # 加载图片 / Load the image
        img = Image.open(image_path)
        img = np.array(img)

        # 检测人脸 / Detect faces
        faces = detector(img, 2)

        if len(faces) == 0:
            return "No face detected", None  # 未检测到人脸

        recognized_names = []
        for face in faces:
            shape = predictor(img, face)
            face_descriptor = face_reco_model.compute_face_descriptor(img, shape)

            # 对比每张人脸特征并寻找最相似的
            min_distance = float("inf")
            min_distance_index = -1

            for i, known_face in enumerate(self.face_feature_known_list):
                distance = self.return_euclidean_distance(face_descriptor, known_face)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_index = i

            # 判断是否匹配某张已知人脸
            if min_distance < 0.4:  # 设置相似度阈值
                name = self.face_name_known_list[min_distance_index]
                recognized_names.append(name)
            else:
                recognized_names.append("Unknown")

        return recognized_names, len(faces)


# 主函数
def main():
    logging.basicConfig(level=logging.INFO)

    # 实例化人脸识别类
    recognizer = FaceRecognizer()

    # 加载人脸数据库
    if not recognizer.get_face_database():
        logging.error("No face database available. Please ensure 'features_all.csv' exists.")
        return

    # 输入文件夹路径
    folder_path = "E:/face_photos/1"

    if not os.path.exists(folder_path):
        logging.error(f"Folder path '{folder_path}' does not exist.")
        return

    # 初始化统计变量
    total_images = 0
    recognized_as_target = 0
    not_recognized_as_target = 0
    target_name = "李书印"  # 目标名字

    # 遍历文件夹下的所有图片
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                total_images += 1

                # 识别图片中的人脸
                recognized_names, face_count = recognizer.recognize_faces_from_image(image_path)

                # 输出识别结果
                if recognized_names == "No face detected":
                    print(f"[{file}] No face detected.")
                    not_recognized_as_target += 1
                else:
                    print(f"[{file}] Detected faces: {recognized_names}")
                    if target_name in recognized_names:
                        recognized_as_target += 1
                    else:
                        not_recognized_as_target += 1

    # 计算准确率
    if total_images > 0:
        accuracy = (recognized_as_target / total_images) * 100
        print("\n=== Summary ===")
        print(f"Total images: {total_images}")
        print(f"Recognized as '{target_name}': {recognized_as_target}")
        print(f"Not recognized as '{target_name}': {not_recognized_as_target}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No valid images found in the specified folder.")


if __name__ == '__main__':
    main()
