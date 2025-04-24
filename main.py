"""main.py – Sistema Inteligente de Contagem de Carros em Semáforo

Executa o pipeline completo:
1. Seleção interativa de ROI (semáforo) e linha virtual.
2. Detecção de veículos com YOLO.
3. Rastreamento com SORT.
4. Contagem condicionada ao sinal verde.
5. Geração de relatórios CSV via report_utils.

Uso:
    python main.py --video samples/video4.mp4

Autor: <Matheus Júlio Boncsidai de Oliveira>
"""

import cv2
import glob
import numpy as np
from sort import Sort 
from config_rect import *
import report_utils

VIDEO_PATH = "samples/video4.mp4"   # caminho padrão (pode vir via argparse)
MARGIN      = 3                      # histerese em pixels
FRAME_SKIP  = 2                      # roda YOLO a cada N frames
FPS_FALLBACK = 30                    # usado se não obtiver FPS do vídeo

# Inicializa o SORT
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Classe para registrar e salvar os dados: report_utils.TrafficReport(FPS) 
report = report_utils.TrafficReport(FPS_FALLBACK)

def draw_bboxes (x, y, h, w, frame, class_name, obj_id=None):
    label = f"{class_name}"
    if obj_id is not None:
        label += f" ID:{int(obj_id)}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def load_model(cfg, weights):
    with open("yolo_models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return classes, net

def detect_vehicles(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Pegar os nomes das camadas de saída
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Forward devolve o resultado da inferência
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
      
    indices = cv2.dnn.NMSBoxes(boxes, 
                               confidences, 
                               score_threshold=0.5, 
                               nms_threshold=0.4)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
    
    if final_boxes:
        return np.array(final_boxes, dtype=float)
    else:
        # Retorna um array vazio; útil para manter inferência do tracker
        return np.empty((0, 4), dtype=float)

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

def detect_light(frame, roi):  
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # máscaras simples 
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

def check_intersection(track, frame, roi, line, track_state, total_count, counted_ids, idx):
    x1, y1, x2, y2, track_id = track
    w = x2 - x1
    h = y2 - y1
    draw_bboxes(x1, y1, w, h, frame, "vehicle",  track_id)

    # Posição atual do veículo
    center_x = (x1+x2)//2 
    center_y = (y1+y2)//2
    pos = side_of_line(center_x,center_y, line)
    last_pos = track_state.get(track_id) 

    # checagem do sinal
    light = detect_light(frame, roi)

    # checagem de cruzamento
    if (last_pos is not None):
        cross_AtoB = last_pos >  MARGIN and pos <= -MARGIN    # sentido +
        cross_BtoA = last_pos < -MARGIN and pos >=  MARGIN    # sentido –
        # Checa cruzamento, se o ID já cruzou, cor do semáforo
        if ((cross_AtoB or cross_BtoA) and 
            (track_id not in counted_ids) and 
            (light == "green")):
            total_count += 1
            counted_ids.add(track_id)
            report.log(idx,track_id, light)

    #Atualiza a última posição de acordo com o id
    track_state[track_id] = pos

    return track_state, counted_ids, total_count

def run_traffic_counter(path):
    #Variáveis para contagem de carros
    counted_ids = set()  # IDs de carros já contados
    total_count = 0
    track_state = {}     #  Estado: passou ou não a linha
    line = None     

    #MAIN
    for idx, frame in enumerate(frame_generator(path)):
        #Configurando linha e retângulo
        if(idx == 0):
            roi, line = select_roi_and_line(frame)

        if idx % FRAME_SKIP == 0:         
            detections = detect_vehicles(frame)

        tracks = tracker.update(detections)
        
        # Loop para contagem de carros
        for track in tracks.astype(int):
            track_state, counted_ids, total_count = check_intersection(track, frame, 
                                                                    roi, line, 
                                                                    track_state, total_count, 
                                                                    counted_ids, idx)
        # Interface: linha de cruzamento
        #            quantidade de carros contados
        cv2.line(frame, line[:2], line[2:], (255, 0, 0), 2)
        cv2.putText(frame, f"Count: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            
        cv2.imshow("Detecção com Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Carregar modelo
classes, net = load_model('yolo_models/yolov4-csp-swish.cfg','yolo_models/yolov4-csp-swish.weights')

run_traffic_counter(VIDEO_PATH)


report.save()
cv2.destroyAllWindows()
