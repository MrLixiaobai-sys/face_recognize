

import cv2
import numpy as np

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
            # similarity_str = "{:.2f}%".format(similarity_percentage)

            # 标注识别结果
            if confidence < 40:
                name = names[label - 1] if label - 1 < len(names) else "未知"
                cv2.putText(frame, f"{name} (similarity:{similarity_str})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 0, 0), 2)
                # cv2.putText(frame, name+" (" + similarity_str + ")", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "未知", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Camera Face Recognition", frame)

        # 按 `空格` 键退出
        if cv2.waitKey(1) & 0xFF == ord(' '):
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