import cv2
import numpy as np

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

    # 遍历筛选出的边框
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 绘制边框和标签
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
