"""web app: tissue from histology photos (upload or live camera) and MoA from BBBC021-like fields

python app.py          # then open http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cellvs.histo import TissuePredictor, prepare

tissue = TissuePredictor(ROOT / "models/tissue_clf.npz")
moa = None
if (ROOT / "data/metadata.csv").exists() and (ROOT / "results/emb_dinov2_vits14.npy").exists():
    from cellvs.moa import MoAPredictor

    moa = MoAPredictor(ROOT / "data/metadata.csv", "dinov2_vits14")


def label_tissue(rgb):
    if rgb is None:
        return None, None
    seen = prepare(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    p = tissue.probs([seen])[0]
    return seen, {tissue.names[i]: float(p[i]) for i in np.argsort(-p)[:5]}


def label_moa(rgb):
    if rgb is None or moa is None:
        return None, None
    r = moa.predict([[rgb[..., 2], rgb[..., 1], rgb[..., 0]]])[0]  # blue=dna, green=tubulin, red=actin
    table = pd.DataFrame(r["vizinhos"]).round({"similaridade": 4})
    head = f"### {r['moa']}\nsimilaridade {r['similaridade']:.4f} (limite para responder: {r['limite']:.4f})"
    return head, table


def results_table(path, cols=None):
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return (df[cols] if cols else df).round(3)


with gr.Blocks(title="cellvs") as demo:
    gr.Markdown("# cellvs\nembeddings de modelos pré-treinados para imagens de células e tecidos. "
                "nada aqui substitui um patologista: são sugestões, com o acerto medido abaixo.")
    with gr.Tab("Tecido (histologia)"):
        gr.Markdown(f"Foto de lâmina, pode ser pelo celular na ocular: o círculo é recortado sozinho. "
                    f"{len(tissue.names)} tecidos.")
        with gr.Row():
            inp = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="foto")
            seen = gr.Image(label="o que o modelo vê", interactive=False)
            out = gr.Label(num_top_classes=5, label="tecido")
        inp.change(label_tissue, inp, [seen, out])
    with gr.Tab("Tecido ao vivo"):
        gr.Markdown("Aponte a câmera para a ocular. Uma resposta por segundo.")
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="câmera")
            live = gr.Label(num_top_classes=3, label="tecido")
        cam.stream(lambda x: label_tissue(x)[1], cam, live, stream_every=1.0, concurrency_limit=1)
    with gr.Tab("MoA (BBBC021)"):
        if moa is None:
            gr.Markdown("Precisa de `data/metadata.csv` e `results/emb_dinov2_vits14.npy`: veja o README.")
        else:
            gr.Markdown("Campo de MCF-7 com 3 canais: azul = DNA, verde = tubulina, vermelho = actina. "
                        f"Biblioteca: {len(moa.lib_t)} tratamentos do BBBC021.")
            with gr.Row():
                minp = gr.Image(sources=["upload"], type="numpy", label="campo")
                with gr.Column():
                    mhead = gr.Markdown()
                    mtab = gr.Dataframe(label="tratamentos mais parecidos")
            minp.change(label_moa, minp, [mhead, mtab])
    with gr.Tab("Quanto acerta"):
        gr.Markdown("**Tecido**: imagens do Wikimedia Commons e recortes de lâminas do GTEx. "
                    "Cada teste deixa autores (Commons) ou doadores (GTEx) inteiros de fora.")
        gr.Dataframe(results_table("results/tecidos_eval.csv", ["teste", "metodo", "n", "top1", "top1_balanceada", "top3", "acaso_top1"]))
        gr.Markdown("**MoA**: BBBC021, vizinho mais próximo ignorando o mesmo composto (NSC) e também a mesma semana (NSCB).")
        gr.Dataframe(results_table("results/holdout_summary.csv"))

if __name__ == "__main__":
    demo.launch()
