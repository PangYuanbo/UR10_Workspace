import cv2
import numpy as np
import matplotlib.pyplot as plt

# YOLOv4-tiny 配置文件、权重文件和类别标签文件的路径
yolo_cfg = "model/yolov4.cfg"
yolo_weights = "model/yolov4.weights"
yolo_names = "model/obj.names"

# 读取类别名称
with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 加载 YOLOv4-tiny 模型
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# 使用 OpenCV 的 DNN 模块设置 YOLO 的后端和目标
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 如果有 GPU，可以设置为 DNN_TARGET_CUDA

# 获取 YOLO 输出层名称
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 检测函数
def detect_with_yolo(image):
    height, width = image.shape[:2]

    # YOLOv4-tiny 需要的图像预处理
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 进行前向传播，获得 YOLO 网络的输出
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # 遍历 YOLO 的输出，提取边框、置信度和类别
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 只考虑置信度大于某个阈值的检测
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                # 获得边框左上角坐标
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 使用非极大值抑制 (NMS) 去除重复边框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 初始化用于绘制的副本图像
    detected_image = image.copy()

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 截取 YOLO 检测到的棋盘区域
            chessboard_img = detected_image[y:y + h, x:x + w].copy()

            # 对棋盘图像进行灰度化和自适应阈值处理
            chessboard_img_gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
            chessboard_img_thresh = cv2.adaptiveThreshold(chessboard_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3)

            # 查找轮廓
            contours, _ = cv2.findContours(chessboard_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            # 绘制最大轮廓
            cv2.drawContours(chessboard_img, [largest_contour], -1, (0, 255, 0), 2)

            # 计算轮廓的近似多边形并找到外角点
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

            # 绘制外角点
            for p in approx:
                cv2.circle(chessboard_img, tuple(p[0]), 5, (255, 0, 0), -1)

            # 将处理后的图像显示出来
            plt.imshow(cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected Chessboard with Contours and Corners")
            plt.show()

            # 绘制 YOLO 边框和标签
            color = (0, 255, 0)  # 绿色框
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(detected_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(detected_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detected_image

# 捕获摄像头图像
cam = cv2.VideoCapture(0)
result, image = cam.read()

if result:
    # 使用 YOLOv4-tiny 检测棋盘
    detected_image = detect_with_yolo(image)

    # 显示 YOLOv4-tiny 检测结果
    cv2.imshow("YOLOv4-tiny Detected Chessboard", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 释放摄像头
cam.release()
