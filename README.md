# cellvs

![tests](https://github.com/richprograming/cellvs/actions/workflows/tests.yml/badge.svg)

O que dá para ler de imagens de microscopia usando só modelos pré-treinados e abertos, com o acerto medido com cuidado para não se enganar com vazamento.

O projeto tem duas partes:

| | Pergunta | Dados | Melhor resultado (em dados nunca vistos) |
|---|---|---|---|
| **MoA** | qual o mecanismo de ação de um composto, a partir de células tratadas? | BBBC021: MCF-7, 38 compostos, 12 MoA | 83/101 tratamentos (NSCB, DINOv2), acaso 0.058 |
| **Tecido** | qual tecido está numa foto de lâmina de histologia? | 316 imagens do Wikimedia Commons + 638 lâminas do GTEx, 29 tecidos | Commons: 180/316 no 1º palpite, 261/316 entre 3, acaso 0.037 |

E um app (`python app.py`) que responde uma foto enviada ou a câmera ao vivo, inclusive foto de celular tirada pela ocular do microscópio.

## Começando

```bash
pip install -r requirements.txt
python app.py               # http://127.0.0.1:7860
pytest
```

Sem os dados baixados, o app ainda responde tecido só pelo zero-shot. As seções abaixo mostram como gerar tudo.

## Parte 1: mecanismo de ação (BBBC021)

Embeddings de ResNet-50 (ImageNet) e DINOv2 ViT-S/14, um por canal (DNA, tubulina, actina), média por tratamento (composto, concentração, semana) e vizinho mais próximo pelo cosseno. **Nenhum treino com os rótulos de MoA.**

- **NSC**: o vizinho mais próximo não pode ser do mesmo composto.
- **NSCB**: também não pode ser da mesma semana (batch). É a métrica que mede se o modelo generaliza ou só reconhece o batch.

| backbone | correção | NSC | NSCB |
|---|---|---|---|
| resnet50 | nenhuma | 0.9196428571428571 (103/112) | 0.7524752475247525 (76/101) |
| resnet50 | centrar no DMSO | 0.8303571428571429 (93/112) | 0.6534653465346535 (66/101) |
| dinov2_vits14 | nenhuma | 0.9553571428571429 (107/112) | 0.8217821782178217 (83/101) |
| dinov2_vits14 | centrar no DMSO | 0.9017857142857143 (101/112) | 0.7524752475247525 (76/101) |

Acaso: NSC 0.059621745874934885, NSCB 0.05825815025531709. O NSCB ignora 11 tratamentos: Cholesterol-lowering e Kinase inhibitors só existem em uma semana.

**Testes de vazamento** (`scripts/3_leak_checks.py`, `results/leak_checks.csv`): prever o MoA só pela semana dá NSC 0.115; o padrão do DMSO de cada placa dá NSCB próximo do acaso; embaralhar o MoA dentro de cada semana derruba o NSCB para 0.13–0.17. Não apareceu vazamento pela semana. Há intervalos de confiança por bootstrap sobre compostos.

**Resposta à pergunta original** (o batch effect pesa mais que o modelo?): centrar no DMSO piorou os dois modelos, e o DINOv2 ficou à frente nas duas condições. Num teste exploratório, feito depois de ver esses números, *sphering* com o DMSO mudou o NSCB mais do que a troca de modelo (resnet50 89/101, dinov2 94/101). Com 38 compostos e intervalos largos, é indício, não conclusão.

**Imagens novas** (`scripts/predict.py`, `scripts/5_holdout.py`): deixando cada semana inteira fora da biblioteca, o DINOv2 acerta o MoA de 1702/2416 fotos individuais. A resposta "desconhecido" funciona mal: quando a classe inteira sai da biblioteca, ele só percebe em 161/2528 fotos. Só faz sentido para imagens feitas como as do BBBC021.

UMAPs por MoA e por semana em `notebooks/01_umap.ipynb` e `results/umap_*.png`.

```bash
# metadados em https://bbbc.broadinstitute.org/BBBC021 (BBBC021_v1_image.csv e BBBC021_v1_moa.csv em data/raw)
python scripts/0_prepare_metadata.py --raw data/raw
python scripts/download_images.py --meta data/metadata.csv --raw data/raw   # só as 3848 imagens usadas, 29 GB, retoma se parar
python scripts/1_embed.py --img-root data/raw --backbone dinov2_vits14      # ~24 min em CPU (Ryzen 5 4600G)
python scripts/1_embed.py --img-root data/raw --backbone resnet50
python scripts/2_evaluate.py --emb results/emb_dinov2_vits14.npy [--no-center]
python scripts/3_leak_checks.py
python scripts/5_holdout.py
python scripts/predict.py --input pasta/ [--watch]                          # azul = DNA, verde = tubulina, vermelho = actina
python scripts/4_build_demo.py                                              # demo/index.html, apresentação interativa (inclui tecidos se houver resultados)
```

Para GPU, há `notebooks/00_colab_embeddings.ipynb` (ainda não testado no Colab).

## Parte 2: tecido em fotos de histologia

![acerto por tecido](results/tecidos_por_tecido.png)

**Dados.** Busquei imagens de 30 tecidos no Wikimedia Commons (`scripts/fetch_commons.py`, que guarda autor e licença de cada uma). A busca traz muito lixo, como esquemas, imuno-histoquímica, tecido doente e fotos que nada têm a ver. Olhei as 1096 imagens e fiquei com 316 de 27 tecidos: H&E ou parecido, tecido normal, foto de microscopia de verdade. A lista está em `scripts/commons_curadoria.csv`. Osso, córtex cerebral e nervo ficaram com menos de 5 imagens boas.

Depois entrou o **GTEx** (`scripts/fetch_gtex.py`): lâminas H&E de tecido normal de doadores, abertas no NCI Imaging Data Commons com licença CC BY 4.0. Peguei 29 lâminas por tecido, cada uma de um doador diferente, em 22 tecidos que batem com a lista (638 lâminas, 464 doadores). Só o nível de ~2 µm/px de cada lâmina é baixado (~30 MB). Dele saem até 8 recortes de 2 mm e 1 mm, parecidos com o campo das objetivas de 10× e 20×, onde pelo menos 60% é tecido, e a lâmina é apagada. São 5059 recortes, 581 MB. Com o GTEx, nervo e córtex cerebral voltaram: 29 tecidos. Osso, cartilagem, ducto deferente, língua, linfonodo, medula espinal, traqueia e ureter não existem no GTEx e seguem só com o Commons. O rótulo do GTEx é o local da coleta, não o que aparece no recorte: um recorte de bexiga pode ser só parede muscular.

**Método.** QuiltNet-B-32, um CLIP treinado com 1 milhão de imagens de patologia, gera um embedding da imagem inteira e dos seus 4 quadrantes (média). A resposta soma duas coisas:
- **zero-shot**: similaridade com frases que descrevem cada tecido ("thyroid with round follicles lined by cuboidal cells and filled with pink colloid"), escritas uma vez a partir de livro-texto;
- **classificador**: regressão logística sobre o mesmo embedding, treinada nas imagens do Commons e nos recortes do GTEx. Cada lâmina do GTEx pesa como uma imagem do Commons, não importa quantos recortes deu.

Fotos tiradas pela ocular têm o círculo iluminado recortado automaticamente.

**Avaliação.** Validação cruzada em 5 partes que deixa **autores inteiros** (Commons) ou **doadores inteiros** (GTEx) de fora: as imagens de um mesmo autor ou doador costumam vir da mesma lâmina, e deixá-las dos dois lados inflaria o resultado. As comparações foram fixadas antes de ver qualquer resultado do GTEx.

| teste | método | 1º palpite | balanceado | entre 3 |
|---|---|---|---|---|
| Commons | zero-shot, sem treino | 159/316 (0.503) | 0.476 | 264/316 (0.835) |
| Commons | zero-shot + classificador, treino só no Commons | 177/316 (0.560) | 0.540 | 270/316 (0.854) |
| Commons | **zero-shot + classificador, treino Commons + GTEx (final)** | **180/316 (0.570)** | **0.521** | **261/316 (0.826)** |
| GTEx | zero-shot, sem treino | 2121/5059 (0.419) | 0.417 | 3348/5059 (0.662) |
| GTEx | zero-shot + classificador, treino só no Commons | 2162/5059 (0.427) | 0.425 | 3105/5059 (0.614) |
| GTEx | **zero-shot + classificador, treino Commons + GTEx (final)** | **2957/5059 (0.585)** | **0.584** | **3956/5059 (0.782)** |
| 8 fotos de celular pela ocular (outras lâminas) | treino só no Commons | 5/8 | | 6/8 |
| 8 fotos de celular pela ocular (outras lâminas) | **treino Commons + GTEx (final)** | **5/8** | | **6/8** |

Acaso: 1/27 = 0.037 no Commons, 1/22 = 0.045 no GTEx. Números completos em `results/tecidos_eval.csv` e `results/tecidos_por_tecido.csv`.

**O GTEx ajudou?** Nas lâminas escaneadas, muito: de 0.427 para 0.585 no 1º palpite. Nas imagens do Commons, quase nada no 1º palpite (+3 imagens) e piorou entre 3 (−9 imagens). Nas fotos do laboratório, nada: os mesmos 5 acertos. Fiquei com o modelo novo porque ele cobre 2 tecidos a mais e empata no resto, mas o ganho não chega ao tipo de foto que mais importa aqui. Duas explicações prováveis: o rótulo do GTEx é o local de coleta (bexiga e intestino delgado acertam 1% e 3% no GTEx, provavelmente porque os recortes são da parede muscular), e uma lâmina escaneada não se parece com uma foto pela ocular.

**O que não funcionou**, medido do mesmo jeito:
- PLIP e BiomedCLIP no zero-shot. O BiomedCLIP chamou cerebelo de cartilagem com 0.94 de probabilidade.
- QuiltNet-B-16, o modelo maior: 0.453 a 0.484 contra 0.494 a 0.560 do B-32.
- Phikon-v2, um ViT-L treinado com lâminas de patologia: sozinho 0.544, junto com o QuiltNet 0.554, e 3/8 e 4/8 nas fotos de celular.
- Vizinho mais próximo pelo DINOv2: 0.295.
- O filtro automático de imagens pelo próprio CLIP: deixou passar muito lixo, por isso a curadoria foi manual.

**Ressalvas.** A configuração do zero-shot e do classificador é a melhor de 8 combinações testadas nas imagens do Commons, então lá o acerto real deve ser um pouco menor. As fotos do laboratório são só 8, de 6 tecidos. Artéria, medula espinal, ureter, intestino delgado e ovário ainda erram muito no 1º palpite.

```bash
python scripts/fetch_commons.py                 # ~1100 imagens, ~330 MB, data/commons + creditos.csv
python scripts/fetch_gtex.py --gb 20            # ~640 lâminas, ~30 min, só os recortes ficam: data/gtex, ~580 MB
python scripts/tissue_train.py                  # avaliação + models/tissue_clf.npz, ~30 min em CPU na 1ª vez
python scripts/tissue_figure.py
python scripts/tissue.py --input fotos/ [--watch]
```

Fotos suas, uma pasta por tecido em `data/tecidos/<tecido>/`, servem de teste externo e depois entram no treino final.

## App

`python app.py` abre quatro abas:

- **Tecido**: envio de foto, colar ou webcam. Mostra o que o modelo viu e os 5 tecidos mais prováveis.
- **Tecido ao vivo**: a câmera é analisada uma vez por segundo.
- **MoA**: um campo de 3 canais, com os tratamentos mais parecidos do BBBC021.
- **Quanto acerta**: as tabelas de resultado.

## Estrutura

```
cellvs/        evaluate.py (NSC/NSCB), embed.py, library.py, moa.py, histo.py
scripts/       0-5 pipeline do BBBC021, predict.py, fetch_commons.py, tissue_*.py
notebooks/     Colab para embeddings, UMAP
tests/         pytest, roda no GitHub Actions
results/       métricas (csv/json) e figuras
app.py         interface Gradio
notes.md       diário: o que funcionou, o que deu errado, bugs
```

## Dados e licenças

O código é MIT. Nenhuma imagem entra no repositório.

- **BBBC021**: imagens e rótulos de AstraZeneca/Broad Institute. Cite Caie et al., *Molecular Cancer Therapeutics*, 2010, e Ljosa et al., *Nature Methods*, 2012.
- **Wikimedia Commons**: cada imagem mantém sua licença (CC BY, CC BY-SA ou domínio público), com autor e link em `data/commons/creditos.csv`. As imagens são baixadas por `fetch_commons.py`.
- **Modelos**: QuiltNet (Ikezogwo et al., *Quilt-1M*, NeurIPS 2023), DINOv2 (Oquab et al., 2023), ResNet-50 do torchvision.

## Referências
- Ljosa et al., *Annotated high-throughput microscopy image sets for validation*, Nature Methods 2012 (BBBC021)
- Perakis et al., *Contrastive Learning of Single-Cell Phenotypic Representations for Treatment Classification*, 2021
- WS-DINO: *Self-Supervised Learning of Phenotypic Representations from Cell Images with Weak Labels*, 2022
- Ikezogwo et al., *Quilt-1M: One Million Image-Text Pairs for Histopathology*, NeurIPS 2023
