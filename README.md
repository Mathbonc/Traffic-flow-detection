# Sistema Inteligente de Contagem de Carros em Semáforo

> **Propósito** — Contar veículos que atravessam um ponto de interseção, respeitando o estado do semáforo (verde/vermelho), a partir de qualquer vídeo de câmera fixa.

---

## ● Demonstração

[![Watch the demo](https://img.youtube.com/vi/0omgnuImMUI/0.jpg)](https://youtu.be/0omgnuImMUI)
1. Seleção do semáforo (retângulo verde) e da linha virtual (linha azul).
2. Rastreamento das bounding‑boxes (retângulos) e incremento do contador.
3. Geração dos CSVs de relatório.

```text
📁 docs/
    └── demo.gif   # <‑ substitua e o banner lá em cima será atualizado
```

---

## ● Pré‑requisitos

| Pacote | Versão testada |
|--------|----------------|
| Python | ≥ 3.9 |
| OpenCV | 4.9 |
| NumPy  | 1.26 |
| pandas | 2.2 |
| filterpy | 1.4 |

Instale tudo de uma vez:
```bash
pip install -r requirements.txt
```

> **Peso dos pesos (`.weights`)** — baixe o modelo **`yolov4-csp-swish`** e coloque em `yolo_models/` (ou use `yolov4‑tiny` se preferir velocidade).

---

## ● Como rodar

```bash
python main.py 
```

* Os relatórios são salvos em **`out/`** automaticamente:
  * `crossings.csv`  • cada veículo cruzado
  * `resumo.csv`     • totais e métricas
  * `distribuicao.csv` • veículos/minuto

---
## ● Variáveis configuráveis
  * `VIDEO_PATH` • Caminho relativo do vídeo a ser utlizado       
  * `FRAME_SKIP` • Analisa um a cada "X" frames                          
  * `FPS_FALLBACK` • FPS do vídeo (padrão = 30)                     

## ● Selecionando Semáforo e Linha

1. **Primeiro frame** é pausado e aberto em uma janela chamada **“Configuração”**.
2. **Botão esquerdo do mouse** — clique e arraste para desenhar o **retângulo** (ROI) que engloba a lâmpada do semáforo.
3. **Botão direito do mouse** — clique no ponto de início e arraste até o ponto final para definir a **linha virtual**.
4. Ajustou tudo? **Pressione `ENTER`**. O vídeo começa a rodar e o contador aparece no canto superior‑esquerdo.

⚠️ Se errar, feche a janela e rode o script de novo.


---

## ● Estrutura do projeto

```text
.
├── main.py             # pipeline principal
├── config_rect.py      # UI de seleção ROI/linha
├── sort.py             # rastreador SORT + Kalman
├── report_utils.py     # geração de CSVs
├── yolo_models/        # pesos .cfg/.weights + coco.names
└── out/                # relatórios gerados
```

---

## ● Ideias de próximos passos

* Auto‑calibração HSV do semáforo.
* Dashboard Matplotlib para leitura direta dos CSVs.
* Exportar resultados como GIF anotado.

---


