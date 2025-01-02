import os


def rename_files(folder_path):
    """
    将文件夹中的文件重命名为数字.姓名.扩展名的格式。
    :param folder_path: 文件夹路径
    """
    try:
        # 获取文件夹内的所有文件
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # 按文件名排序（确保一致性）
        file_list.sort()

        for index, file_name in enumerate(file_list, start=1):
            # 分离文件名和扩展名
            base_name, ext = os.path.splitext(file_name)

            # 获取姓名部分（文件名前的部分）
            name = base_name.split('.')[0]

            # 构造新文件名
            new_name = f"{index}.{name}{ext}"

            # 构造完整路径
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {old_path} -> {new_path}")

    except Exception as e:
        print(f"重命名过程中发生错误: {e}")


# 文件夹路径
folder_path = r"E:\人脸识别数据\train_data\YM"
rename_files(folder_path)
