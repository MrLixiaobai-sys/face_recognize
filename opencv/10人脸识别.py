import cv2
import numpy as np
from PIL import Image
import os


def recognize_face(image_path, model_path, names):
    """
    使用训练好的模型识别单张图片中的人脸。
    :param image_path: 待识别的图片路径
    :param model_path: 训练模型路径
    :param names: 人员名称列表，对应标签
    """
    # 加载训练好的模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # 加载人脸检测器
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # 读取图片并转为灰度
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像

    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    for x, y, w, h in faces:
        roi = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi)

        # 标注识别结果
        if confidence < 80:
            name = names[label - 1] if label - 1 < len(names) else "未知"
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        else:
            cv2.putText(img, "未知", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 测试图片路径
    image_path = r"E:\face_recognize\recogn_data\KA\1.tiff"

    # 训练模型路径
    model_path = "trainer/trainer.yml"

    # 定义标签对应的人员名称
    names = ["KA", "KL", "KM", "KR", "MK", "NA", "NM", "TM", "UY", "YM"]  # 根据你的数据更新

    # 执行识别
    recognize_face(image_path, model_path, names)