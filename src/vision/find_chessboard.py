import cv2
import numpy as np

def find_chessboard(image, roi=None):
    if roi:
        # 使用给定的 ROI 进行裁剪
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
    else:
        print("未提供ROI，检测整张图像")

    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(blurred, 10, 100)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一张副本图像，用来绘制符合条件的轮廓
    filtered_contours_image = image.copy()

    # 打印所有轮廓的面积，帮助调试
    print(f"检测到的轮廓数量: {len(contours)}")

    # 遍历所有轮廓并过滤面积在 20,000 到 35,000 之间的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20000 < area < 35000:
            print(f"绘制轮廓，面积: {area}")
            # 绘制面积符合条件的轮廓
            cv2.drawContours(filtered_contours_image, [contour], -1, (0, 255, 0), 2)  # 绿色边框

    # 显示符合条件的轮廓
    cv2.imshow("Filtered Contours (Area 20k-35k)", filtered_contours_image)

    # 等待用户按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 捕获摄像头图像
cam = cv2.VideoCapture(0)
result, image = cam.read()

if result:
    # 使用你提供的ROI坐标和大小
    roi = (450, 240, 300, 300)  # 这是棋盘的近似位置

    # 调用 find_chessboard 函数检测并绘制符合条件的轮廓
    find_chessboard(image, roi)

# 释放摄像头
cam.release()
