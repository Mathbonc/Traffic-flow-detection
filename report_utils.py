# report_utils.py
import csv, os, pandas as pd

class TrafficReport:
    def __init__(self, fps, out_dir="out"):
        self.fps   = fps
        self.events = []                     # (frame, sec, id, light)
        self.out   = out_dir
        os.makedirs(self.out, exist_ok=True)

    def log(self, frame_idx, track_id, light_state):
        """Chame quando contar um veículo."""
        self.events.append((
            frame_idx,
            frame_idx / self.fps,
            int(track_id),
            light_state
        ))

    def save(self):
        if not self.events:
            print("Nenhum veículo contabilizado.")
            return

        # 1) cruzamentos
        cross_file = os.path.join(self.out, "crossings.csv")
        with open(cross_file, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "sec", "id", "light"])
            wr.writerows(self.events)

        # 2) resumo
        total      = len(self.events)
        total_time = self.events[-1][1]

        df = pd.DataFrame(self.events, columns=["frame","sec","id","light"])
        cont_light = (df["light"].value_counts()
                        .reindex(["green","red"])
                        .fillna(0)
                        .astype(int))

        resumo_file = os.path.join(self.out, "resumo.csv")
        with open(resumo_file, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([
                "total_veiculos", "duracao_s", "fps",
                "veic_por_min", "veic_por_seg",
                "veic_sinal_verde", "veic_sinal_vermelho"
            ])
            wr.writerow([
                total,
                round(total_time, 2),
                self.fps,
                round(total / (total_time/60), 2),
                round(total / total_time, 3),
                int(cont_light["green"]),
                int(cont_light["red"])
            ])
            
        # 3) distribuição temporal (1 min)
        df = pd.DataFrame(self.events,
                          columns=["frame","sec","id","light"])
        df["bucket_min"] = (df["sec"] // 60).astype(int)
        dist = (df.groupby("bucket_min")
                  .size()
                  .rename("veiculos")
                  .reset_index())
        dist.to_csv(os.path.join(self.out, "distribuicao.csv"),
                    index=False)

        print("✔ Relatórios salvos em", self.out)
