import cv2
import os

# 设置保存路径
save_path = r"D:\face_recognize\recognize_git\recognition_image_list"
if not os.path.exists(save_path):
    os.makedirs(save_path)  # 如果文件夹不存在，创建该文件夹

# 设置文件名
name = "lishu.jpg"
save_file = os.path.join(save_path, name)

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 捕获摄像头图像
    ret, frame = video_capture.read()

    # 显示摄像头图像
    cv2.imshow("Press 's' to save, 'q' to quit", frame)

    # 按下 's' 保存图片
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(save_file, frame)
        print(f"照片已保存到: {save_file}")
        break  # 保存后退出

    # 按下 'q' 退出
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        print("退出程序")
        break

# 释放摄像头资源
video_capture.release()
cv2.destroyAllWindows()
