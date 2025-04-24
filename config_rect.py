import cv2

def select_roi_and_line(frame, win_name="Configuração"):
    """
    Permite ao usuário marcar um retângulo (ROI) e uma linha
    sobre `frame` com o mouse.
    Retorna: (roi_rect), (count_line)
    """
    clone = frame.copy()
    roi_rect = None        # (x, y, w, h)
    line_pts = []          # [(x1,y1), (x2,y2)]
    drawing_roi   = False
    drawing_line  = False
    x0 = y0 = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal clone, roi_rect, line_pts, drawing_roi, drawing_line, x0, y0

        # --- Retângulo: clique com BOTÃO ESQUERDO ---
        if event == cv2.EVENT_LBUTTONDOWN and not roi_rect:
            drawing_roi = True
            x0, y0 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing_roi:
            img = frame.copy()
            cv2.rectangle(img, (x0, y0), (x, y), (0, 255, 0), 2)
            cv2.imshow(win_name, img)

        elif event == cv2.EVENT_LBUTTONUP and drawing_roi:
            drawing_roi = False
            x1, y1 = x, y
            x, y   = min(x0, x1), min(y0, y1)
            w, h   = abs(x1 - x0), abs(y1 - y0)
            roi_rect = (x, y, w, h)
            clone = frame.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

        # --- Linha: clique com BOTÃO DIREITO ---
        if event == cv2.EVENT_RBUTTONDOWN and len(line_pts) == 0:
            drawing_line = True
            line_pts.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and drawing_line:
            img = clone.copy()
            cv2.line(img, line_pts[0], (x, y), (255, 0, 0), 2)
            cv2.imshow(win_name, img)

        elif event == cv2.EVENT_RBUTTONUP and drawing_line:
            drawing_line = False
            line_pts.append((x, y))
            cv2.line(clone, line_pts[0], line_pts[1], (255, 0, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_cb)
    cv2.imshow(win_name, frame)

    print("➜  Arraste **botão esquerdo** p/ ROI; **botão direito** p/ linha; ENTER p/ finalizar.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):        # ENTER
            break
    cv2.destroyWindow(win_name)

    if roi_rect is None or len(line_pts) != 2:
        raise RuntimeError("Seleção incompleta – tente novamente.")
    (x1, y1), (x2, y2) = line_pts
    return roi_rect, (x1, y1, x2, y2)
