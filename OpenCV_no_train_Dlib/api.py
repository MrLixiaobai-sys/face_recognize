import dlib
import numpy as np
import cv2
import time
import logging
from PIL import Image, ImageDraw

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")



# 处理获取的视频流，进行人脸识别 / Face detection and recognition from input video stream
def process(self, stream):
    # 1. 读取存放所有人脸特征的 csv / Read known faces from "features.all.csv"
    if self.get_face_database():
        while stream.isOpened():
            self.frame_cnt += 1
            logging.debug("Frame %d starts", self.frame_cnt)
            flag, img_rd = stream.read()
            faces = detector(img_rd, 0)
            kk = cv2.waitKey(1)
            # 按下 q 键退出 / Press 'q' to quit
            if kk == ord('q'):
                break
            else:
                draw_note(self,img_rd)
                self.current_frame_face_feature_list = []
                self.current_frame_face_cnt = 0
                self.current_frame_face_name_position_list = []
                self.current_frame_face_name_list = []
                self.current_frame_face_similarity_list = []  # 清空相似度列表

                # 2. 检测到人脸 / Face detected in current frame
                if len(faces) != 0:
                    # 3. 获取当前捕获到的图像的所有人脸的特征 / Compute the face descriptors for faces in current frame
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape))
                    # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                    for k in range(len(faces)):
                        logging.debug("For face %d in camera:", k + 1)
                        # 先默认所有人不认识，是 unknown / Set the default names of faces with "unknown"
                        self.current_frame_face_name_list.append("unknown")

                        # 每个捕获人脸的名字坐标 / Positions of faces captured
                        self.current_frame_face_name_position_list.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 5. 对于某张人脸，遍历所有存储的人脸特征
                        current_frame_e_distance_list = []
                        for i in range(len(self.face_feature_known_list)):
                            # 如果 person_X 数据不为空
                            if str(self.face_feature_known_list[i][0]) != '0.0':
                                e_distance_tmp = self.return_euclidean_distance(
                                    self.current_frame_face_feature_list[k], self.face_feature_known_list[i])
                                logging.debug("  With person %s, the e-distance is %f", str(i + 1), e_distance_tmp)
                                current_frame_e_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                current_frame_e_distance_list.append(999999999)

                        # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e-distance
                        similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                        similarity = min(current_frame_e_distance_list)  # 获取最小的欧式距离作为相似度

                        logging.debug("Minimum e-distance with %s: %f",
                                      self.face_name_known_list[similar_person_num], similarity)

                        if similarity < 0.3:  # 假设相似度阈值是 0.4
                            self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                            self.current_frame_face_similarity_list.append(similarity)  # 保存相似度
                            logging.debug("Face recognition result: %s",
                                          self.face_name_known_list[similar_person_num])
                        else:
                            self.current_frame_face_name_list[k] = "Unknown"
                            self.current_frame_face_similarity_list.append(similarity)  # 保存相似度
                            logging.debug("Face recognition result: Unknown person")

                        logging.debug("\n")

                        # 矩形框 / Draw rectangle
                        for kk, d in enumerate(faces):
                            # 绘制矩形框
                            cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                          (255, 255, 255), 2)

                    self.current_frame_face_cnt = len(faces)

                    # 8. 写名字 / Draw name
                    img_with_name = draw_name(self,img_rd)

                else:
                    img_with_name = img_rd

            logging.debug("Faces in camera now: %s", self.current_frame_face_name_list)

            cv2.imshow("camera", img_with_name)

            # 9. 更新 FPS / Update stream FPS
            self.update_fps()
            logging.debug("Frame ends\n\n")


# 更新 FPS / Update FPS of Video stream
def update_fps(self):
    now = time.time()
    # 每秒刷新 fps / Refresh fps per second
    if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
        self.fps_show = self.fps
    self.start_time = now
    self.frame_time = now - self.frame_start_time
    self.fps = 1.0 / self.frame_time
    self.frame_start_time = now


# 绘制名字和相似度
def draw_name(self, img_rd):
    # 在人脸框下面写人脸名字 / Write names under rectangle
    img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for i in range(self.current_frame_face_cnt):
        # 强制将 name 转换为字符串，防止 numpy.float64 类型的错误
        name = str(self.current_frame_face_name_list[i])
        similarity = self.current_frame_face_similarity_list[i]  # 获取当前人脸的相似度



        # 将相似度转化为百分比，并保留两位小数
        similarity_percentage = (1 - similarity) * 100  # 欧式距离越小，相似度越高
        similarity_text = f"Similarity: {similarity_percentage:.2f}%"  # 格式化成百分比形式

        # 绘制名字
        name_position = self.current_frame_face_name_position_list[i]
        # 往上调整，比如上移 10 像素
        adjusted_position = (name_position[0], name_position[1] - 300)
        if name == "Unknown":
            draw.text(
                xy=adjusted_position,
                text=name,
                font=self.font_chinese,
                fill=(255, 0, 0)
            )
        else:
            draw.text(
                xy=adjusted_position,
                text=name,
                font=self.font_chinese,
                fill=(0, 255, 0)
            )


        # 绘制相似度
        if name != "Unknown":
            position = (
                self.current_frame_face_name_position_list[i][0],
                self.current_frame_face_name_position_list[i][1] - 330)
            draw.text(
                xy=position,
                text=similarity_text,
                font=self.font_chinese,
                fill=(0, 255, 0)
            )

        img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img_rd

 # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window
def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps_show.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


