import os
import time

import cv2
import numpy as np
from PIL import Image

"""
数据集结构要求如下:
E:\face_recognize\train_data
│
├── 1
│   ├── 1.KA.tiff
│   ├── 2.KA.tiff
│   ├── 3.KA.tiff
│   └── ...
│
├── 2
│   ├── 1.KB.tiff
│   ├── 2.KB.tiff
│   ├── 3.KB.tiff
│   └── ...
│
└── 3...
"""

"""
遍历所有子文件夹，获取人脸图像和对应的标签。
:param base_path: 数据集主目录路径
:return: faces: 人脸图像列表, ids: 对应的标签列表
"""
# 记录训练开始时间
start_time = time.time()

def get_images_and_labels(base_path):
    faces = []
    ids = []

    # Haar特征分类器

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # 遍历所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 跳过非文件夹项

        # 遍历文件夹中的所有图片
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # 读取图片并转为灰度图
            PIL_img = Image.open(file_path).convert('L')
            # 将灰度图转换为NumPy数组，这样可以使用OpenCV进行处理
            img_numpy = np.array(PIL_img, 'uint8')

            # 提取人脸区域
            # 使用加载的Haar特征分类器检测图像中的人脸，返回检测到的人脸的坐标和尺寸（x, y, w, h)
            faces_detected = face_detector.detectMultiScale(img_numpy)
            for x, y, w, h in faces_detected:
                faces.append(img_numpy[y:y + h, x:x + w])  # 裁剪人脸区域
                ids.append(int(folder_name))  # 使用文件夹名作为标签

    return faces, ids


if __name__ == "__main__":
    # 数据集路径
    # base_path = r"E:\face_recognize\train_data"
    base_path = r"E:\face_recognize\200张数据\train_data"
    # 获取图像和标签
    faces, ids = get_images_and_labels(base_path)

    # 创建人脸识别模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()



    # 人脸识别训练,id:姓名对应一个faces:人脸数据
    recognizer.train(faces, np.array(ids))

    # 记录训练结束时间
    end_time = time.time()

    training_time = end_time - start_time
    print(f"训练完成，耗时：{training_time:.2f} 秒")

    # 保存模型
    model_path = "trainer/trainer.yml"
    recognizer.write(model_path)
    print(f"模型已保存至: {model_path}")
