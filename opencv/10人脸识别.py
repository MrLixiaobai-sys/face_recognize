# import cv2
# import numpy as np
# from PIL import Image
# import os
#
#
# def recognize_face(image_path, model_path, names):
#     """
#     使用训练好的模型识别单张图片中的人脸。
#     :param image_path: 待识别的图片路径
#     :param model_path: 训练模型路径
#     :param names: 人员名称列表，对应标签
#     """
#     # 加载训练好的模型
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(model_path)
#
#     # 加载人脸检测器
#     face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
#
#     # 读取图片并转为灰度
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
#
#     # 检测人脸
#     faces = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
#     for x, y, w, h in faces:
#         roi = gray[y:y + h, x:x + w]
#         label, confidence = recognizer.predict(roi)
#
#         # 标注识别结果
#         if confidence < 80:
#             name = names[label - 1] if label - 1 < len(names) else "未知"
#             cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
#         else:
#             cv2.putText(img, "未知", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # 显示结果
#     cv2.imshow("Result", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     # 测试图片路径
#     image_path = r"E:\face_recognize\recogn_data\KA\1.tiff"
#
#     # 训练模型路径
#     model_path = "trainer/trainer.yml"
#
#     # 定义标签对应的人员名称
#     names = ["KA", "KL", "KM", "KR", "MK", "NA", "NM", "TM", "UY", "YM"]  # 根据你的数据更新
#
#     # 执行识别
#     recognize_face(image_path, model_path, names)


# import cv2
# import numpy as np
#
#
#
# def recognize_face_from_camera(model_path, names):
#     """
#     使用训练好的模型实时识别摄像头中的人脸。
#     :param model_path: 训练模型路径
#     :param names: 人员名称列表，对应标签
#     """
#     # 加载训练好的模型
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(model_path)
#
#     # 加载人脸检测器
#     face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
#
#     # 打开摄像头
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Cannot access the camera.")
#         return
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Cannot read the frame from camera.")
#             break
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
#
#         # 检测人脸
#         faces = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
#
#         for x, y, w, h in faces:
#             roi = gray[y:y + h, x:x + w]
#             label, confidence = recognizer.predict(roi)
#
#             # 计算相似度，这里假设最大置信度为1000（可按需调整），将其换算为百分比形式
#             similarity_percentage = (1 - confidence / 100) * 100
#             similarity_str = f"{similarity_percentage:.2f}%"  # 保留两位小数，输出例如 "70.00%"
#             # similarity_str = "{:.2f}%".format(similarity_percentage)
#
#             # 标注识别结果
#             if confidence < 40:
#                 name = names[label - 1] if label - 1 < len(names) else "未知"
#                 cv2.putText(frame, f"{name} (similarity:{similarity_str})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                             (255, 0, 0), 2)
#                 # cv2.putText(frame, name+" (" + similarity_str + ")", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
#             else:
#                 cv2.putText(frame, "未知", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         # 显示结果
#         cv2.imshow("Camera Face Recognition", frame)
#
#         # 按 `空格` 键退出
#         if cv2.waitKey(1) & 0xFF == ord(' '):
#             break
#
#     # 释放摄像头资源
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     # 训练模型路径
#     model_path = "trainer/trainer.yml"
#
#     # 定义标签对应的人员名称
#     names = ["李书印", "zebin", "KM", "KR", "MK", "NA", "NM", "TM", "UY", "YM"]  # 根据你的数据更新
#
#     # 执行识别
#     recognize_face_from_camera(model_path, names)
#

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def recognize_face_from_camera(model_path, names):
    """
    使用训练好的模型实时识别摄像头中的人脸。
    :param model_path: 训练模型路径
    :param names: 人员名称列表，对应标签
    """
    # 加载训练好的模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # 加载人脸检测器
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read the frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图像

        # 检测人脸
        faces = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))

        for x, y, w, h in faces:
            roi = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(roi)

            # 计算相似度，这里假设最大置信度为1000（可按需调整），将其换算为百分比形式
            similarity_percentage = (1 - confidence / 100) * 100
            similarity_str = f"{similarity_percentage:.2f}%"  # 保留两位小数，输出例如 "70.00%"

            # 标注识别结果
            if confidence < 40:
                name = names[label - 1] if label - 1 < len(names) else "未知"
                # 使用 PIL 绘制中文文本
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)  # 选择中文字体文件
                draw.text((x, y - 30), f"{name} (相似度:{similarity_str})", font=font, fill=(255, 0, 0))
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 将 PIL 图像转换回 OpenCV 图像
            else:
                cv2.putText(frame, "未知", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # 绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Camera Face Recognition", frame)

        # 按 `q` 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 训练模型路径
    model_path = "trainer/trainer.yml"

    # 定义标签对应的人员名称
    names = ["李书印", "zebin", "KM", "KR", "MK", "NA", "NM", "TM", "UY", "YM"]  # 根据你的数据更新

    # 执行识别
    recognize_face_from_camera(model_path, names)
