import cv2
import glob

DAIR_V2X_PATH = 'samples/DAIR-V2X-C/*.jpg'
VIDEO1_PATH = 'samples/video1.mp4'
VIDEO2_PATH = 'samples/video2.mp4'

def draw_boxes (h,w, frame, class_name, detection):
    #Desenha a bounding box
    center_x = int(detection[0] * w)
    center_y = int(detection[1] * h)
    width = int(detection[2] * w)
    height = int(detection[3] * h)

    x = int(center_x - width / 2)
    y = int(center_y - height / 2)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def load_model (cfg, weights):
# Carregar nomes das classes (80 classes COCO)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Carregar modelo
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # ou CUDA se tiver
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # ou DNN_TARGET_CUDA
    
    return classes, net

def detect_frame (frame):
    # Criar blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Pegar os nomes das camadas de saída
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Forward recebe o resultado da inferência
    detections = net.forward(output_layers)

    h, w = frame.shape[:2]
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            # Checa se o objeto detectado faz parte da classe de veículos e se a confiança é decente.
            if classes[class_id] in ["car", "truck", "bus", "motorbike"] and confidence > 0.5:
                class_name = classes[class_id]
                draw_boxes(h,w,frame,class_name, detection)
            
    cv2.imshow("Detecção", frame)

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
classes, net = load_model('yolov4-tiny.cfg','yolov4-tiny.weights')

    
for frame in frame_generator(VIDEO1_PATH):
    detect_frame(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
