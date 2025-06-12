import cv2
import os


def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0  # 用于计数当前帧
    while True:
        ret, frame = cap.read()
        if not ret:  # 如果没有读取到帧，结束循环
            break

            # 设置输出图片的文件名
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # 保存当前帧为图片
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()  # 释放视频捕获对象
    print(f"提取完成，总共提取了 {frame_count} 帧.")


# 使用示例
video_path = r'C:\Users\25677\Desktop\sp\6.mp4'  # 替换为你的视频文件路径
output_folder = r'C:\Users\25677\Desktop\photos\data\12'  # 替换为你想保存图片的文件夹
extract_frames(video_path, output_folder)