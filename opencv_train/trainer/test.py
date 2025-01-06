import os

# 定义根目录路径
root_dir = r"E:\face_recognize\500人7000张"

# 遍历根目录下的每个子目录（每个子目录对应一个人）
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)

    # 确保是目录且符合命名规则（假设目录名是001, 002, ...）
    if os.path.isdir(subdir_path) and subdir.isdigit():
        # 遍历该目录下的所有图片
        for img_file in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_file)

            # 确保是图片文件
            if os.path.isfile(img_path) and img_file.endswith(".jpg"):
                # 获取图片的基础名称（如02.jpg）
                base_name, ext = os.path.splitext(img_file)

                # 生成新的文件名：保持原文件名的数字部分，加上文件夹名作为 `shu` 后缀
                new_name = f"{base_name}.shu{subdir}.jpg"
                new_path = os.path.join(subdir_path, new_name)

                # 重命名图片文件
                os.rename(img_path, new_path)
                print(f"已重命名：{img_path} -> {new_path}")

print("所有文件处理完成！")
