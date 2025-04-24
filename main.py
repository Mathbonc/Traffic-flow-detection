import cv2
import glob
import numpy as np
from sort import Sort 
from config_rect import *

DAIR_V2X_PATH = 'samples/DAIR-V2X-C/*.jpg'
VIDEO1_PATH = 'samples/video1.mp4'
VIDEO2_PATH = 'samples/video2.mp4'
VIDEO3_PATH = 'samples/video3.mp4'
VIDEO4_PATH = 'samples/video4.mp4'
VIDEO5_PATH = 'samples/video5.mp4'

# Inicializa o SORT
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
vehicle_ids = set()

#Variáveis para contagem de carros
# Variáveis para detectar a passagem de carros da linha
prev_centroids = {}     # id → y do frame anterior
counted_ids    = set()  # ids já contados
total_count    = 0
track_state = {}

# Variáveis de frameskip
frame_skip   = 2        # ⇒ processa 1 de cada 3 quadros

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

def detect_light(frame, roi):  # x,y,w,h
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # máscaras simples (ajuste para seu vídeo):
    red1 = cv2.inRange(hsv, (0,100,100), (10,255,255))
    red2 = cv2.inRange(hsv, (160,100,100), (180,255,255))
    green = cv2.inRange(hsv, (40, 50, 50), (90,255,255))

    red_px   = cv2.countNonZero(red1 | red2)
    green_px = cv2.countNonZero(green)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return "green" if green_px > red_px else "red"

def side_of_line(px, py, line):
    if line is None:
        raise ValueError("Parâmetro 'line' é None – defina (x1,y1,x2,y2) antes de chamar.")
    x1, y1, x2, y2 = line
    return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)


# Carregar modelo
classes, net = load_model('yolo_models/yolov4-csp-swish.cfg','yolo_models/yolov4-csp-swish.weights')

line = None 

#MAIN
for idx, frame in enumerate(frame_generator(DAIR_V2X_PATH)):
    #Configurando linha e retângulo
    if(idx == 0):
        roi, line = select_roi_and_line(frame)

    if idx % frame_skip == 0:         
        detections = detect_vehicles(frame)

    tracks = tracker.update(detections)
    
    # Checando o cruzamento
    for track in tracks.astype(int):
        x1, y1, x2, y2, track_id = track
        w = x2 - x1
        h = y2 - y1
        draw_boxes(x1, y1, w, h, frame, "vehicle",  track_id)

        # definindo o lado do veículo
        center_x = (x1+x2)//2 #centro atual
        center_y = (y1+y2)//2 #centro atual
 
        pos = side_of_line(center_x,center_y, line)
        last = track_state.get(track_id) # ultima posição


        # checagem do sinal
        light = detect_light(frame, roi)
        print(pos)

        # checagem de cruzamento
        if last is not None and last > 0 and pos <= 0 and track_id not in counted_ids:
            total_count += 1
            counted_ids.add(track_id)
            print(f"Veículo {track_id} contado! Total = {total_count}")


        #Atualiza a posição 
        track_state[track_id] = pos

    #printa a linha
    cv2.line(frame, line[:2], line[2:], (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        
    cv2.imshow("Detecção com Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
