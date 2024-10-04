import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 使用摄像头捕获图像
cam = cv2.VideoCapture(0)
result, frame_orig = cam.read()

# 释放摄像头
cam.release()

# 检查摄像头是否成功捕获图像
if not result:
    print("Error: Unable to capture image from camera")
else:
    # 显示从摄像头捕获的图像
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB))
    plt.title('Original frame from camera')
    plt.show()


    # YOLO 检测部分（您可以复用之前的 YOLO 检测代码）
    def detect_with_yolo(image):
        height, width = image.shape[:2]

        # YOLOv4-tiny 需要的图像预处理
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # 进行前向传播，获得 YOLO 网络的输出
        layer_outputs = net.forward(output_layers)

        boxes = []
        confidences = []

        # 遍历 YOLO 的输出，提取边框和置信度
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

        # 使用非极大值抑制 (NMS) 去除重复边框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 初始化用于绘制的副本图像
        detected_image = image.copy()
        selected_box = None

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # 绘制 YOLO 边框
                cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                selected_box = (x, y, w, h)

        return detected_image, selected_box


    # 调用 YOLO 检测
    detected_image, box = detect_with_yolo(frame_orig)

    # 显示 YOLO 检测结果
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.title('YOLO Bounding Box')
    plt.show()

    # 如果成功检测到棋盘，执行进一步的图像处理
    if box is not None:
        x, y, w, h = box
        img = frame_orig[y:y + h, x:x + w].copy()

        # 缩放图像并进行 Canny 边缘检测
        d = 128
        frame = cv2.resize(img, (d, d), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]

        frame = cv2.Canny(frame, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 显示缩放后的图像和 Canny 边缘检测结果
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Frame Downscaled + Canny Edge Detection')
        plt.show()

        # 绘制边框
        cv2.rectangle(frame, (0, 0), (w, h), 255, 1)
        cv2.rectangle(frame, (0, 0), (w, 10), 255, -1)
        cv2.putText(frame, 'board', (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # 显示最终处理后的图像
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image with YOLO Bounding Box')
        plt.show()
    else:
        print("未检测到棋盘区域")
