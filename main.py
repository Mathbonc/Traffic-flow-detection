import cv2
import glob
import numpy as np
from sort import Sort 

DAIR_V2X_PATH = 'samples/DAIR-V2X-C/*.jpg'
VIDEO1_PATH = 'samples/video1.mp4'
VIDEO2_PATH = 'samples/video2.mp4'

# Inicializa o SORT
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
vehicle_ids = set()

#Variáveis para contagem de carros
# Variáveis para detectar a passagem de carros da linha
prev_centroids = {}     # id → y do frame anterior
counted_ids    = set()  # ids já contados
total_count    = 0
track_state = {}

# Variáveis de frameskip
frame_skip   = 3        # ⇒ processa 1 de cada 3 quadros

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

empty = np.empty((0, 5))
line_y = None 

for idx, frame in enumerate(frame_generator(VIDEO1_PATH)):
    # Definindo a linha horizontal
    if line_y is None:        
        line_y = int(frame.shape[0] * 0.55)

    if idx % frame_skip == 0:          # roda YOLO
        detections = detect_vehicles(frame)
    if detections.size:            # anexa coluna "score"
        detections = np.hstack((detections,
                                    np.ones((detections.shape[0], 1))))
    else:                              # só predição
        detections = empty

    tracks = tracker.update(detections)
    
    for track in tracks.astype(int):
        x1, y1, x2, y2, track_id = track
        w = x2 - x1
        h = y2 - y1
        draw_boxes(x1, y1, w, h, frame, "vehicle",  track_id)

        #checando se passou da linha
        center_y = (y1+y2)//2 #centro atual
        prev_center = prev_centroids.get(track_id) #centros anteriores

        if prev_center is not None:
            # Exemplo: conta quando vem de cima (prev < line) e passa para baixo (cy ≥ line)
            if (prev_center < line_y) and (center_y>=line_y):     #Se o anterior estava antes do treshold e o atual depois, cruzou a linha
                if track_id not in counted_ids:    
                    total_count += 1
                    counted_ids.add(track_id)
                    print(f"Veículo {track_id} contado! Total = {total_count}")


        #Atualiza a posição do centro
        prev_centroids[track_id] = center_y

    #printa a linha
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        
    cv2.imshow("Detecção com Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
