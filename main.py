import cv2
import glob
import numpy as np
from sort import Sort 

DAIR_V2X_PATH = 'samples/DAIR-V2X-C/*.jpg'
VIDEO1_PATH = 'samples/video1.mp4'
VIDEO2_PATH = 'samples/video2.mp4'

# Inicializa o SORT
tracker = Sort(max_age=10, min_hits=3)
vehicle_ids = set()

def draw_boxes (x, y, h, w, frame, class_name, obj_id=None):
    #Desenha a bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"{class_name}"
    if obj_id is not None:
        label += f" ID:{int(obj_id)}"

    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def load_model(cfg, weights):
    with open("yolo_models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return classes, net

def detect_vehicles(frame):
    # Criar blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Pegar os nomes das camadas de saída
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Forward recebe o resultado da inferência
    detections = net.forward(output_layers)
  
    boxes = []
    confidences = []

    h, w = frame.shape[:2]
    for output in detections:
        for detection in output:

            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if classes[class_id] in ["car", "truck", "bus", "motorbike"] and confidence > 0.5:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, x + bw, y + bh])
                confidences.append(float(confidence))

            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
    
    return np.array(final_boxes)

def frame_generator(input_path):
    # Caso seja uma pasta de imagens ( *.jpg)
    if "*" in input_path:
        image_files = glob.glob(input_path)
        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is not None:
                yield frame

    # Caso seja um vídeo
    elif input_path.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
    else:
        raise ValueError("Formato de entrada não suportado.")

# Carregar modelo
classes, net = load_model('yolo_models/yolov4-csp-swish.cfg','yolo_models/yolov4-csp-swish.weights')

frame_skip = 1
frame_count = 0

for frame in frame_generator(DAIR_V2X_PATH):
    if frame_count % frame_skip == 0:
        detections = detect_vehicles(frame)
        tracks = tracker.update(detections)
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            w = int(x2 - x1)
            h = int(y2 - y1)
            draw_boxes(int(x1), int(y1), w, h, frame, "vehicle",  track_id)
            
        cv2.imshow("Detecção com Tracking", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
