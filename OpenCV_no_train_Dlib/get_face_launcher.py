import subprocess
import time


# 启动 get_faces_from_camera_tkinter.py 文件
def start_get_faces_from_camera():
    print("启动 get_faces_from_camera_tkinter.py...")
    process = subprocess.Popen(["python", "get_faces_from_camera_tkinter.py"])
    process.wait()  # 等待窗口关闭
    print("get_faces_from_camera_tkinter.py 已关闭。")
    return process.returncode


# 启动 features_extraction_to_csv.py 文件
def start_features_extraction():
    print("启动 features_extraction_to_csv.py...")
    process = subprocess.Popen(["python", "features_extraction_to_csv.py"])
    process.wait()  # 等待脚本执行完成
    print("features_extraction_to_csv.py 执行完成。")
    return process.returncode


def main():
    # 启动第一个脚本并等待其完成
    ret_code = start_get_faces_from_camera()

    # 如果第一个脚本正常退出，则启动第二个脚本
    if ret_code == 0:
        time.sleep(1)  # 等待1秒确保资源释放
        start_features_extraction()
    else:
        print("get_faces_from_camera_tkinter.py 未正常退出，终止操作。")


if __name__ == "__main__":
    main()
