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

    # 初始化变量存储最大轮廓
    max_contour = None
    max_area = 0

    # 打印所有轮廓的面积，帮助调试
    print(f"检测到的轮廓数量: {len(contours)}")

    # 遍历所有轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        print(f"轮廓面积: {area}")  # 打印每个轮廓的面积用于调试

        # 过滤面积在20000到40000之间的轮廓
        if 20000 < area < 40000:  # 根据实际面积调整范围
            # 近似轮廓为多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 打印近似多边形的顶点数
            print(f"近似多边形顶点数: {len(approx)}")

            # 如果近似的多边形有四个顶点，说明是一个矩形
            if len(approx) == 4 and area > max_area:
                max_contour = approx
                max_area = area

    # 如果找到合适的轮廓，绘制在原图上
    if max_contour is not None:
        # 绘制检测到的四边形（棋盘外边框）
        cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 3)

        # 返回裁剪后的图像和棋盘轮廓
        return image, max_contour
    else:
        print("未能检测到棋盘")
        return image, None

# 捕获摄像头图像
cam = cv2.VideoCapture(0)
result, image = cam.read()

if result:
    # 使用你提供的ROI坐标和大小
    roi = (450, 240, 300, 300)  # 这是棋盘的近似位置

    # 调用 find_chessboard 函数检测棋盘
    detected_image, chessboard_contour = find_chessboard(image, roi)

    # 显示结果图像
    if detected_image is not None:
        cv2.imshow("Detected Chessboard", detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 释放摄像头
cam.release()
