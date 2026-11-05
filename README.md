# cellvs

Prever o mecanismo de ação (MoA) de compostos a partir de imagens de células (BBBC021, Cell Painting-like), usando embeddings de modelos pré-treinados e **sem treinar com os rótulos de MoA**.

Pergunta principal: o quanto a correção de *batch effect* pesa mais do que a escolha do modelo?

## Status
Em andamento. Código de dados, embeddings e avaliação pronto. Resultados ainda não rodados (veja `notes.md`).

| backbone | correção | NSC | NSCB |
|---|---|---|---|
| resnet50 | nenhuma | – | – |
| resnet50 | centrar no DMSO | – | – |
| dinov2_vits14 | nenhuma | – | – |
| dinov2_vits14 | centrar no DMSO | – | – |

## Rodando
```bash
pip install -r requirements.txt

# dados: https://bbbc.broadinstitute.org/BBBC021  (imagens + BBBC021_v1_image.csv e BBBC021_v1_moa.csv)
python scripts/0_prepare_metadata.py --raw data/raw
python scripts/1_embed.py --img-root data/raw --backbone resnet50
python scripts/2_evaluate.py --emb results/emb_resnet50.npy
python scripts/2_evaluate.py --emb results/emb_resnet50.npy --no-center

pytest
```

## Métricas
- **NSC**: vizinho mais próximo (cosseno) de cada tratamento, ignorando o mesmo composto.
- **NSCB**: igual, mas ignorando também o mesmo batch (semana). É a que mede se o modelo generaliza ou só memoriza o batch.

## Referências
- Perakis et al., *Contrastive Learning of Single-Cell Phenotypic Representations for Treatment Classification* (2021)
- WS-DINO: *Self-Supervised Learning of Phenotypic Representations from Cell Images with Weak Labels* (2022)
- Ljosa et al., *Annotated high-throughput microscopy image sets for validation* (BBBC021)
