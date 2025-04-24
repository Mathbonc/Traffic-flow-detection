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

VIDEO_PATH = "samples/DAIR-V2X-C/*.jpg"   # caminho padrão (pode vir via argparse)
MARGIN      = 3                      # histerese em pixels
FRAME_SKIP  = 2                      # roda YOLO a cada N frames
FPS_FALLBACK = 30                    # usado se não obtiver FPS do vídeo

# Inicializa o SORT
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
classes, net = None, None  # definidos em load_model()

# Classe para registrar e salvar os dados: report_utils.TrafficReport(FPS) 
report = report_utils.TrafficReport(FPS_FALLBACK)

def draw_bboxes(x: int, y: int, h: int, w: int, frame: np.ndarray, class_name: str,
                obj_id: int | None = None) -> None:
    """Desenha bounding‑box + rótulo.

    Parameters
    ----------
    x, y, h, w
        Coordenadas/cotas da caixa no formato OpenCV.
    frame
        Frame BGR sobre o qual desenhar.
    class_name
        Nome da classe (ex.: "vehicle").
    obj_id
        ID opcional atribuído pelo rastreador; se fornecido, é adicionado ao
        rótulo.
    """
    label = class_name if obj_id is None else f"{class_name} ID:{obj_id}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)

def load_model(cfg: str, weights: str) -> tuple[list[str], cv2.dnn_Net]:
    """Carrega a rede YOLO a partir de arquivos *cfg* e *weights*.

    Returns
    -------
    classes
        Lista de rótulos disponíveis no modelo.
    net
        Objeto `cv2.dnn_Net` pronto para inferência.
    """
    with open("yolo_models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return classes, net

def detect_vehicles(frame: np.ndarray) -> np.ndarray:
    """Executa YOLO e devolve bounding‑boxes de veículos.

    Se não houver detecções, devolve um array vazio `(0, 4)` — isso mantém
    o SORT vivo apenas em modo predição.
    """
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, 
                                crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    detections = net.forward(output_layers)   # Forward devolve o resultado da inferência
  
    boxes, confidences = [], []
    h, w = frame.shape[:2]
    
    for output in detections:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            if classes[class_id] not in {"car", "truck", "bus", "motorbike"}:
                continue
            confidence = scores[class_id]
            if confidence < 0.5:
                continue
            cx, cy, bw, bh = det[:4]
            cx, cy = int(cx * w), int(cy * h)
            bw, bh = int(bw * w), int(bh * h)
            x, y = int(cx - bw / 2), int(cy - bh / 2)
            boxes.append([x, y, x + bw, y + bh])
            confidences.append(float(confidence))
      
    indices = cv2.dnn.NMSBoxes(boxes, 
                               confidences, 
                               score_threshold=0.5, 
                               nms_threshold=0.4)

    final = [boxes[i] for i in indices.flatten()] if len(indices) else []
    return np.asarray(final, dtype=float) if final else np.empty((0, 4), dtype=float)

def frame_generator(path: str):
    """Itera sobre frames de um vídeo ou lista de imagens.

    Parameters
    ----------
    path
        Caminho para vídeo (`.mp4`, `.avi`, ...), ou glob `*.jpg`.
    """
    # Caso seja uma pasta de imagens ( *.jpg)
    if "*" in path:
        image_files = glob.glob(path)
        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is not None:
                yield frame
    # Caso seja um vídeo
    elif path.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
    else:
        raise ValueError("Formato de entrada não suportado.")

def detect_light(frame: np.ndarray, roi: tuple[int, int, int, int]) -> str:  
    """Detecta cor dominante (verde/vermelho) no retângulo do semáforo.

    Parameters
    ----------
    frame
        Nº‑array BGR do frame.
    roi
        Tupla `(x, y, w, h)` com a ROI onde fica a lâmpada.

    Returns
    -------
    "green" ou "red".
    """
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
