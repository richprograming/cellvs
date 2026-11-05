# notes

Working log. Newest at the bottom. Fill this in as I go, including what didn't work.

## goal
Predict mechanism of action (MoA) of compounds from cell images (BBBC021), without training on MoA labels.
Compare ImageNet ResNet-50 vs DINOv2 embeddings, and see how much batch correction matters.

## evaluation
- NSC: 1-NN (cosine) on treatment-level embeddings, neighbor must be a different compound
- NSCB: same, and the neighbor must also come from a different batch (week)
- this is the usual protocol in the BBBC021 papers, so numbers are comparable-ish. Not exactly: they differ in
  how treatments are aggregated, so don't compare against published tables without checking that

## open questions
- aggregation per (compound, conc, week): does that match how others do it?
- DMSO centering is crude. Try TVN (typical variation normalization)
- fine-tune on the cell images or keep backbones frozen?
- 12 MoA classes, some with 2-3 compounds only -> high variance, report per-class too

## log
- [x] get data, run 0_prepare_metadata.py
- [x] resnet50 baseline, raw vs centered
- [x] dinov2_vits14
- [x] UMAP plots in notebooks/

## 2026-11-05: primeira rodada com dados reais

Ambiente: Ryzen 5 4600G, 14 GB de RAM, sem GPU. Python 3.11.9, torch 2.15.0, torchvision 0.29.2, numpy 2.4.6, pandas 3.1.1, opencv 5.0.0.93, scikit-learn 1.9.1, umap-learn 0.5.12.

### renomeação
Pacote, imports e README agora são `cellvs`. A pasta do projeto não renomeou: o VS Code a mantinha aberta e o Windows negou o acesso. O projeto não era um repositório git; criei um, com um commit do estado original antes de mexer em tudo.

### bugs (nenhum destes aparecia nos testes)
- `pytest` puro e `python scripts/2_evaluate.py` falhavam com `ModuleNotFoundError`: a raiz do projeto não estava no `sys.path`. Só `python -m pytest` passava.
- `Image_FileNames_*` não existe. O nome real é `Image_FileName_*`. As outras colunas estavam certas.
- O CSV de MoA tem uma linha `DMSO`, que viraria uma 13ª classe. Tirei, e agora dá 12.
- O batch saía `Week1/Week1`, porque o split era em `_`. Agrupava certo por acaso, mas o rótulo estava errado. Agora é `Week1`.
- `RuntimeError: expected scalar type Double but found Float` em `1_embed.py`: `np.percentile` devolve float64 e o numpy 2 promove a imagem inteira.
- `torch.hub` falhou uma vez com "no internet connection" ao carregar o DINOv2. Na tentativa seguinte funcionou, sem mudar nada. Não descobri a causa.
- libtiff gera vários warnings por imagem, por causa de tags estranhas nos TIFFs. Silenciei com `OPENCV_LOG_LEVEL`. As imagens leem bem: uint16, 1024x1280.

### dados
Metadados: 3848 imagens (2528 rotuladas + 1320 DMSO), 38 compostos, 12 classes, 103 composto-concentrações, 10 semanas, 55 placas. Bate com a página do BBBC.

O dataset inteiro são 55 zips (45,68 GB) que, extraídos, dariam uns 104 GB: mais do que o espaço livre. As 3848 imagens estão espalhadas por todas as placas, então escrevi `download_images.py`, que busca só os arquivos necessários dentro dos zips com HTTP Range. Levou 1072 s para 11544 arquivos (30,3 GB em disco). A primeira versão abria uma conexão por leitura e fazia ~2 s/arquivo; com keep-alive e 4 placas em paralelo caiu para ~0,1 s/arquivo.

### embeddings
resnet50: 1232 s, 6144 dimensões. dinov2_vits14: 1414 s, 1152 dimensões. Sem NaN e sem linhas duplicadas. Não precisou de GPU. O notebook do Colab existe, mas nunca rodou.

Os pesos do resnet50 são `ResNet50_Weights.DEFAULT`, ou seja, IMAGENET1K_V2 (`resnet50-11ad3fa6.pth`).

### resultados
Centralização por semana (o padrão do `2_evaluate.py`):

| backbone | correção | NSC | NSCB |
|---|---|---|---|
| resnet50 | nenhuma | 103/112 | 76/101 |
| resnet50 | centrar no DMSO | 93/112 | 66/101 |
| dinov2_vits14 | nenhuma | 107/112 | 83/101 |
| dinov2_vits14 | centrar no DMSO | 101/112 | 76/101 |

Duas surpresas: 107/112 sem correção nenhuma me pareceu bom demais, e centralizar no DMSO piorou tudo, quando eu esperava o contrário.

### procurando vazamento (`scripts/3_leak_checks.py`)
Comparações reais não vazam: os testes confirmam que o vizinho nunca é o mesmo composto, nem a mesma semana no NSCB. Cada imagem aparece uma vez só. Depois testei se a semana entrega a resposta:

- só a semana, sem imagem (vizinho sorteado na mesma semana): NSC 0.11526269504789435
- só o DMSO da semana, sem nenhum sinal do tratamento: NSCB 0.05814127465378116 (resnet50) e 0.06327632763276328 (dinov2), praticamente o acaso de 0.05825815025531709
- MoA embaralhado entre compostos da mesma semana (200 vezes): NSC entre 0.169063 e 0.189821, NSCB entre 0.134107 e 0.169405, nas 8 variantes

Conclusão: a semana sozinha não explica os números. Eles parecem reais neste protocolo, mas "não achei vazamento" não é "não existe". Não testei a posição do poço na placa, e compostos quimicamente parecidos entre si (análogos) contam como compostos diferentes.

O que pode explicar parte do número alto: a própria página do BBBC diz que este subconjunto foi escolhido por ter um MoA claro, e 6 das 12 classes foram definidas visualmente. O problema é fácil por construção, então uma acurácia alta aqui não vale para compostos em geral.

### incerteza
IC de 95% por bootstrap de compostos (2000 reamostragens) em `results/leak_checks.csv`. São largos, por exemplo NSCB do resnet50 sem correção entre 0.591386 e 0.894118. Diferenças pareadas no NSCB (`results/leak_checks_summary.json`):

- dinov2 - resnet50: IC exclui zero sem correção (0.0118 a 0.1237) e com centralização (0.0204 a 0.1947)
- sem correção - centrado: IC exclui zero no resnet50 (0.0092 a 0.1979), mas não no dinov2 (-0.0680 a 0.1936)

Sem correção para comparações múltiplas e com limite inferior quase em zero: tratar como indício.

A dose de taxol 0.3 está nas 10 semanas e nas 55 placas (parece um controle positivo, o que é inferência minha): são 1332 das 1428 imagens de Microtubule stabilizers, e ela pesa 10 dos 112 tratamentos. Sem ela os números mudam pouco, ver colunas `*_no_taxol_control`.

### por classe (`results/*_per_moa.csv`)
A média esconde muita coisa. Aurora, Eg5, Protein synthesis e DNA damage ficam em 100% no NSCB nas 4 configurações da tabela. Microtubule destabilizers é a mais afetada pela troca de semana: NSC 13/14 em todas, NSCB em 0.14285714285714285 no resnet50 e 0.21428571428571427 no dinov2 sem correção.

A centralização não piora igual: no dinov2 ela sobe destabilizers de 0.21428571428571427 para 0.5714285714285714, e derruba DNA replication de 0.875 para 0.25 e Protein degradation de 0.7142857142857143 para 0.14285714285714285. Classes com n entre 5 e 18: um tratamento pesa muito.

### batch effect (`notebooks/01_umap.ipynb`, `results/batch_vs_moa.csv`)
Existe, mas não domina. No UMAP, os grupos grandes são de MoA e a semana aparece como sub-regiões dentro deles.

- vizinho mais próximo na mesma semana: 0.36607142857142855 sem correção (os dois modelos), contra 0.12369270626840526 ao acaso
- silhouette sem correção: por MoA 0.35319 (resnet50) e 0.42543 (dinov2); por semana -0.10397 e -0.14770
- a queda de NSC para NSCB mostra o mesmo efeito

Estranho: centralizar **aumentou** o vizinho na mesma semana, para 0.4732142857142857 (resnet50) e 0.4375 (dinov2), além de baixar o silhouette por MoA. Não sei por quê. Hipótese não testada: tirar a média comum faz o cosseno olhar para variações pequenas e ruidosas, que ainda carregam a semana.

### exploratório: outras correções
Feito depois de ver a tabela, então não é comparação limpa. Nada foi ajustado olhando o MoA, mas só tentei uma configuração.

| backbone | variante | NSC | NSCB |
|---|---|---|---|
| resnet50 | centrar no DMSO por placa | 91/112 | 67/101 |
| resnet50 | placa + sphering (PCA do DMSO, 99% da variância) | 106/112 | 89/101 |
| dinov2_vits14 | centrar no DMSO por placa | 100/112 | 83/101 |
| dinov2_vits14 | placa + sphering (PCA do DMSO, 99% da variância) | 105/112 | 94/101 |

O sphering foi o melhor NSCB de todas as variantes e aproximou os dois modelos. Precisaria de confirmação em outro conjunto.

### perguntas em aberto
- **Agregar por (composto, concentração, semana) é o que os outros fazem?** Sem resposta completa: não conferi o protocolo nos artigos. O que sei: dá 112 tratamentos contra os 103 composto-concentrações, e a diferença toda vem da dose de taxol em 10 semanas. Continua sem comparar com tabelas publicadas.
- **Centralizar no DMSO é grosseiro demais? TVN ajuda?** Pior do que grosseiro neste setup: piorou por semana e, no resnet50, também por placa. Testei só um sphering; TVN completo (com CORAL) não foi feito.
- **Fine-tuning ou backbones congelados?** Não testei. Tudo aqui é congelado.
- **Classes com poucos compostos dão variância alta?** Sim: por classe em `results/*_per_moa.csv`, intervalos largos em `results/leak_checks.csv`. Cholesterol-lowering e Kinase inhibitors nem entram no NSCB.
- **Nova: por que centralizar aproxima vizinhos da mesma semana?** Em aberto.

### limitações
- O campo inteiro (1024x1280) é espremido para 224x224: perde resolução celular e distorce a proporção. Pode prejudicar classes sutis, e os artigos de referência costumam usar recortes por célula. Não testei.
- Normalização por percentil por imagem: pode apagar parte da diferença de intensidade entre semanas, o que ajudaria a explicar o desempenho sem correção. Não testei.
- Nenhum `requirements` fixado: as versões usadas estão no início desta entrada.
