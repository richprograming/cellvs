# cellvs

Prever o mecanismo de ação (MoA) de compostos a partir de imagens de células (BBBC021, Cell Painting-like), usando embeddings de modelos pré-treinados e **sem treinar com os rótulos de MoA**.

Pergunta principal: o quanto a correção de *batch effect* pesa mais do que a escolha do modelo?

## Status
Rodado com os dados reais: 3848 imagens (2528 rotuladas + 1320 de DMSO), 38 compostos, 12 classes de MoA, 112 tratamentos por (composto, concentração, semana). Detalhes e ressalvas em `notes.md`.

| backbone | correção | NSC | NSCB |
|---|---|---|---|
| resnet50 | nenhuma | 0.9196428571428571 (103/112) | 0.7524752475247525 (76/101) |
| resnet50 | centrar no DMSO | 0.8303571428571429 (93/112) | 0.6534653465346535 (66/101) |
| dinov2_vits14 | nenhuma | 0.9553571428571429 (107/112) | 0.8217821782178217 (83/101) |
| dinov2_vits14 | centrar no DMSO | 0.9017857142857143 (101/112) | 0.7524752475247525 (76/101) |

Acerto esperado ao acaso: NSC 0.059621745874934885, NSCB 0.05825815025531709. O NSCB ignora 11 tratamentos: Cholesterol-lowering e Kinase inhibitors só existem em uma semana.

A centralização é pela média do DMSO de cada semana. Acurácia por classe em `results/*_per_moa.csv`; intervalos, testes de vazamento e variantes em `results/leak_checks.csv`.

Resposta parcial à pergunta principal: a centralização no DMSO piorou o resultado nos dois modelos, e o DINOv2 ficou à frente do ResNet-50 nas duas condições. Em um teste exploratório, feito depois de ver esses números, sphering com o DMSO mudou o NSCB mais do que a troca de modelo (resnet50 89/101, dinov2_vits14 94/101). Com um único conjunto de 38 compostos e intervalos largos, isso é indício, não conclusão.

## Rodando
```bash
pip install -r requirements.txt

# metadados: https://bbbc.broadinstitute.org/BBBC021  (BBBC021_v1_image.csv e BBBC021_v1_moa.csv em data/raw)
python scripts/0_prepare_metadata.py --raw data/raw
# baixa só as 3848 imagens usadas (13 GB transferidos, 30 GB no disco); pode ser repetido se parar
python scripts/download_images.py --meta data/metadata.csv --raw data/raw

python scripts/1_embed.py --img-root data/raw --backbone resnet50
python scripts/1_embed.py --img-root data/raw --backbone dinov2_vits14
python scripts/2_evaluate.py --emb results/emb_resnet50.npy
python scripts/2_evaluate.py --emb results/emb_resnet50.npy --no-center
python scripts/2_evaluate.py --emb results/emb_dinov2_vits14.npy
python scripts/2_evaluate.py --emb results/emb_dinov2_vits14.npy --no-center
python scripts/3_leak_checks.py
python scripts/4_build_demo.py  # gera demo/index.html, com as imagens embutidas (fora do git)

pytest
```

Sem GPU, em um Ryzen 5 4600G, os embeddings levaram cerca de 21 min (resnet50) e 24 min (dinov2_vits14). Para rodar com GPU há `notebooks/00_colab_embeddings.ipynb`, ainda não testado no Colab. Os UMAPs estão em `notebooks/01_umap.ipynb` e `results/umap_*.png`.

## Imagens novas
`scripts/predict.py` sugere o MoA de uma imagem nova pelos tratamentos mais parecidos da biblioteca BBBC021, ou responde "desconhecido" abaixo de um limite de similaridade. Ele aceita um CSV com os 3 canais separados ou uma pasta com uma imagem de 3 canais por campo (azul = DNA, verde = tubulina, vermelho = actina). Com `--watch`, responde cada imagem salva na pasta.

```bash
python scripts/predict.py --input pasta_com_imagens/
python scripts/predict.py --input pasta_com_imagens/ --watch
python scripts/5_holdout.py   # simula experimentos novos: cada semana fora da biblioteca, uma de cada vez
```

Só faz sentido para imagens feitas como as do BBBC021: MCF-7, 24 h com um composto, as mesmas 3 marcações e o mesmo aumento. Em outro laboratório o batch effect é maior, e a biblioteca precisa ser refeita com referências próprias.

Na simulação (`results/holdout_summary.csv`), o DINOv2 acertou o MoA de 1702 de 2416 fotos individuais de compostos que a biblioteca nunca viu. A resposta "desconhecido" funciona mal: quando a classe inteira sai da biblioteca, ele só percebe em 161 de 2528 fotos e nas outras sugere um MoA errado.

## Fotos de histologia (experimental)
`scripts/tissue.py` sugere o tecido de fotos tiradas pela ocular, como lâminas de H&E fotografadas com o celular. Ele recorta o círculo iluminado e dá duas respostas:
- **zero-shot**: o QuiltNet, um CLIP treinado com imagens de patologia, compara a foto com uma frase por tecido. Não precisa de exemplos. Abaixo de 0.2 de probabilidade, a resposta é "incerto".
- **referência**: a foto mais parecida, pelo DINOv2, entre as que você já identificou em `data/tecidos/<tecido>/`. Se a similaridade passar de 0.8, a referência vale mais que o zero-shot.

```bash
python scripts/tissue.py --input data/fotos_lab/
python scripts/tissue.py --input data/fotos_lab/ --watch
```

Foi testado com 9 fotos e sem gabarito: a comparação é com a minha leitura visual (cólon, língua, 2 de cerebelo, medula, órgão tubular, músculo e uma em tricrômico). No zero-shot, ele respondeu 3 fotos (língua, cerebelo e ducto deferente para o órgão tubular) e deu "incerto" nas outras 6. Os dois limites foram escolhidos olhando essas mesmas fotos. Os únicos pares parecidos são da mesma lâmina, então reconhecer o tecido em lâminas diferentes ainda não foi testado. PLIP e BiomedCLIP foram piores: o BiomedCLIP chamou cerebelo de cartilagem com 0.94.

## Métricas
- **NSC**: vizinho mais próximo (cosseno) de cada tratamento, ignorando o mesmo composto.
- **NSCB**: igual, mas ignorando também o mesmo batch (semana). É a que mede se o modelo generaliza ou só memoriza o batch.

## Dados
As imagens e o ground truth têm copyright da AstraZeneca e não entram neste repositório. Citação recomendada pelo BBBC: Caie et al., *Molecular Cancer Therapeutics*, 2010, e Ljosa et al., *Nature Methods*, 2012.

## Referências
- Perakis et al., *Contrastive Learning of Single-Cell Phenotypic Representations for Treatment Classification* (2021)
- WS-DINO: *Self-Supervised Learning of Phenotypic Representations from Cell Images with Weak Labels* (2022)
- Ljosa et al., *Annotated high-throughput microscopy image sets for validation* (BBBC021)
