import cv2
import numpy as np

def find_chessboard(image):
    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化变量存储最大轮廓
    max_contour = None
    max_area = 0

    # 遍历所有轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        # 如果轮廓面积足够大，且形状为一个大致的矩形
        if area > 1000:
            # 近似轮廓为多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 如果近似的多边形有四个顶点，说明是一个矩形
            if len(approx) == 4 and area > max_area:
                max_contour = approx
                max_area = area

    # 如果找到合适的轮廓，绘制在原图上
    if max_contour is not None:
        cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 3)
        return image, max_contour
    else:
        print("未能检测到棋盘")
        return image, None

# 捕获摄像头图像
cam = cv2.VideoCapture(0)
result, image = cam.read()

if result:
    # 调用 find_chessboard 函数检测棋盘
    detected_image, chessboard_contour = find_chessboard(image)

    # 显示结果图像
    cv2.imshow("Detected Chessboard", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 释放摄像头
cam.release()
