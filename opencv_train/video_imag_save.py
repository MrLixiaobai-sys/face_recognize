# # 导入模块
# import cv2
#
# # 摄像头
# cap = cv2.VideoCapture(0)
#
# falg = 1
# num = 41
#
# while (cap.isOpened()):  # 检测是否在开启状态
#     ret_flag, Vshow = cap.read()  # 得到每帧图像
#     if not ret_flag:
#         print("Error: Failed to grab frame!")
#         break
#
#     cv2.imshow("Capture_Test", Vshow)  # 显示图像
#     k = cv2.waitKey(1) & 0xFF  # 按键判断
#     if k == ord('s'):  # 保存
#         save_path = "E:/face_photos/"
#         cv2.imwrite(save_path + str(num) + ".lishu"+".jpg", Vshow)
#         print("success to save" + str(num) + ".jpg")
#         print("-------------------")
#         num += 1
#     elif k == ord(' '):  # 退出
#         break
# # 释放摄像头
# cap.release()
# # 释放内存
# cv2.destroyAllWindows()
import cv2
import time
import os

# 摄像头初始化
cap = cv2.VideoCapture(0)

# 检查摄像头是否打开
if not cap.isOpened():
    print("Error: Cannot open camera!")
    exit()

# 创建保存图片的文件夹
save_path = "E:/face_photos/3"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 每秒保存图片的帧数
fps = 24
# 视频时长（秒）
video_duration = 10
# 当前时间记录
start_time = time.time()

# 图片编号
num = 1

while cap.isOpened():
    ret_flag, frame = cap.read()  # 读取每一帧
    if not ret_flag:
        print("Error: Failed to grab frame!")
        break

    # 显示当前帧
    cv2.imshow("Capture_Test", frame)

    # 计算当前秒数
    elapsed_time = time.time() - start_time

    # 如果已经经过了完整的秒数，保存图片
    if int(elapsed_time * fps) > num - 1:
        filename = os.path.join(save_path, f"{num}.lishu.jpg")
        cv2.imwrite(filename, frame)  # 保存图像
        print(f"Success to save {num}.jpg")
        num += 1

    # 如果视频播放时长超过 5 秒，则退出
    if elapsed_time > video_duration:
        break

    # 按 'q' 键退出
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# 释放摄像头
cap.release()
# 释放内存
cv2.destroyAllWindows()

