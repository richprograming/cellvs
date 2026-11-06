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

## 2026-11-06: demo e imagens novas

### demo
`scripts/4_build_demo.py` gera `demo/index.html`, um arquivo só com as fotos reais, os vizinhos e os resultados. Ele confere que o vizinho mostrado é o mesmo da avaliação, e todos os números saem de `results/`. Fica fora do git porque embute imagens da AstraZeneca. A publicação com link foi bloqueada pelas permissões.

### predict.py
Pergunta: e se chegar uma imagem nova? O `predict.py` compara cada foto com a biblioteca de 103 tratamentos (todas as semanas, sem correção) e devolve os 5 mais parecidos. Abaixo de um limite de similaridade, responde "desconhecido". O limite é o percentil 5 da similaridade que fotos conhecidas atingem com um tratamento de outro composto e outra semana. Ele aceita CSV com 3 canais separados ou um arquivo de 3 canais por foto, e tem modo de vigiar pasta.

Movi `get_backbone` e a leitura de imagem de `1_embed.py` para `cellvs/embed.py`. Para conferir: refiz o embedding de 6 imagens e a diferença para o `.npy` salvo deu 0.0. A mesma imagem como arquivo RGB único ou como 3 arquivos também deu diferença 0.0.

Bug: em `library.threshold`, comparar colunas de texto em 2D quebra no pandas 3 (`StringArray` não faz broadcast). Agora converto para array de objetos do numpy.

Teste de ponta a ponta com a Week10 fora da biblioteca, em 12 fotos (4 de AZ138, 4 de AZ-U, 4 de taxol): 11 acertos, nos modos CSV, pasta e vigiar pasta, com as mesmas respostas. O erro foi um AZ138 (Eg5) visto como destabilizador de microtúbulo, e os dois mecanismos param a mitose. Amostra pequena, serve só para mostrar que funciona.

### simulação de experimento novo (`scripts/5_holdout.py`)
Cada semana sai da biblioteca, uma de cada vez. O composto da consulta também sai, para simular um composto novo.

| backbone | nível | top-1 | top-5 | acaso |
|---|---|---|---|---|
| resnet50 | tratamento | 76/101 | 86/101 | 0.058258 |
| resnet50 | foto | 1525/2416 | 1969/2416 | 0.059263 |
| dinov2_vits14 | tratamento | 83/101 | 90/101 | 0.058258 |
| dinov2_vits14 | foto | 1702/2416 | 2078/2416 | 0.059263 |

Por tratamento dá exatamente o NSCB da tabela, como deveria: é o mesmo protocolo. Por foto individual cai, porque uma foto é mais ruidosa que a média de um tratamento. A média por foto é dominada pelo taxol, que tem 1332 das 2416 fotos. Pesando as classes igualmente, o DINOv2 por foto fica em 0.7358954339764003 (`top1_balanced`). Por classe, em `results/holdout_per_moa.csv`, o DINOv2 vai de 0.35714285714285715 (Microtubule destabilizers) a 0.96875 (Protein synthesis).

**"Desconhecido" funciona mal.** Com a classe inteira fora da biblioteca, a resposta certa é "desconhecido", mas ele só percebe em 161/2528 fotos (DINOv2) e 141/2528 (resnet50). Por tratamento, 23/112 e 17/112. AUROC entre conhecido e classe removida: 0.694248 por foto e 0.82107 por tratamento (DINOv2). Ou seja, um mecanismo novo quase sempre recebe o rótulo de um mecanismo parecido, com similaridade alta. Isso precisa estar claro para qualquer uso real. Hipótese: sem correção, todas as similaridades ficam perto de 0.9 e a margem entre conhecido e novo é pequena.

### fotos do laboratório
As fotos que tenho são de células de tecido, sem tratamento com compostos, com marcação desconhecida. O `predict.py` não serve para elas: a biblioteca é de MCF-7 tratadas. O que faria sentido é busca por semelhança e agrupamento entre as próprias fotos, sem MoA. Ainda não fiz.

Atualização: são 9 fotos de lâminas de histologia tiradas pela ocular com o celular, quase todas em H&E e uma em tricrômico. Não há gabarito, então comparo com a minha leitura visual: cólon, língua, 2 de cerebelo, medula espinal, um órgão tubular (ducto deferente ou ureter?), 2 do mesmo campo de músculo com conjuntivo e a de tricrômico.

- **DINOv2 entre as próprias fotos.** As duas fotos de cerebelo dão 0.860 e as duas do mesmo campo dão 0.907. Fotos sem relação ficam em 0.71 ou menos. Os dois pares parecidos são da mesma lâmina, então isso não mostra que ele reconhece o tecido em lâminas diferentes.
- **Zero-shot, a foto inteira contra uma frase por tecido, com 30 tecidos e 3 frases cada:**
  - PLIP errou bastante: chamou cerebelo de cartilagem.
  - BiomedCLIP acertou o cólon com 0.81, mas chamou cerebelo de cartilagem com 0.94. Errar com essa confiança é o pior caso.
  - QuiltNet acertou língua (0.23) e as 2 de cerebelo (0.24 e 0.14), e o cólon só ficou em 3º. Ele é o menos confiante, o que combina com o que ele sabe.
  - Dividir em 9 pedaços não melhorou de forma clara.
- **Similaridade foto a foto pelo QuiltNet não serve.** Fotos sem relação chegam a 0.91, e o par de cerebelo dá 0.966. Para comparar fotos, ficou o DINOv2.

A primeira versão do `scripts/tissue.py` usava os dois limites, 0.2 no zero-shot e 0.8 na referência. Foi substituída pelo que vem abaixo.

O dono das fotos confirmou os 6 tecidos da minha leitura visual. A de tricrômico ficou sem rótulo.

### tecidos com imagens públicas
Não há mais fotos do laboratório, então busquei imagens públicas.

- **Repositório da UCF (UCF-MultiOrgan-Path).** Tem 977 lâminas de 15 órgãos, mas devolve 403 para download automático e cada órgão tem de 7 a 20 GB. Não tentei contornar o bloqueio.
- **Download do Commons.**
  - A busca pela API de 30 tecidos trouxe 1107 imagens.
  - Bug: o servidor de imagens devolve 429 com uma página HTML para miniaturas de tamanho fora do padrão (1024 px) e para originais. O script salvava essa página como `.jpg`. Agora pede só 960 ou 500 px e confere o content-type.
- **Curadoria.**
  - O filtro automático pelo CLIP, com frases como "H&E" contra "desenho" e "imuno-histoquímica", deixou passar muito lixo. Nessa versão o resultado foi 0.311 no zero-shot e 0.288 no classificador.
  - Montei folhas de miniaturas numeradas por tecido e escolhi à mão: H&E ou próximo, tecido normal, foto real e órgão certo. Sobraram 316 imagens de 27 tecidos (`scripts/commons_curadoria.csv`).
  - Coisas que apareceram na busca: fotos de uma cerimônia militar em "colon", plantas e rostos em "tongue", quase só coloração por prata em "cerebellum".
  - 4 imagens de "colon" eram músculo esquelético e mudaram de classe. 2 imagens apareceram em duas classes e saíram.
- **Avaliação.** GroupKFold em 5 partes por autor, porque a série de um autor costuma ser a mesma lâmina.

| variante (QuiltNet-B-32 salvo indicação) | 1º palpite |
|---|---|
| zero-shot, frases simples, imagem inteira | 0.503 |
| zero-shot + classificador, inteira + 4 quadrantes, frases descritivas | 0.560 (177/316), entre 3: 0.854 |
| QuiltNet-B-16, melhor combinação | 0.484 |
| Phikon-v2 (linear) + zero-shot | 0.544 |
| QuiltNet + Phikon-v2 + zero-shot | 0.554 |
| vizinho mais próximo DINOv2 | 0.295 |

A configuração final é a melhor de 8 testadas nas mesmas imagens, então 0.560 é otimista. As frases descritivas foram escritas uma vez para os 30 tecidos, a partir de livro-texto e sem olhar os erros, mas eu já tinha visto o resultado por tecido da versão anterior.

Nas 8 fotos do laboratório, com o classificador treinado só no Commons, o 1º palpite acertou 5/8: os 2 de cerebelo, cólon, ducto deferente e língua. A medula foi para o 2º lugar, e as 2 de músculo esquelético erraram. Depois do teste, as fotos entram no treino final do `models/tissue_clf.npz`.

O Hibou-B não foi testado: precisa de `trust_remote_code`, ou seja, rodar código do repositório do modelo, e isso foi bloqueado.

O app (`app.py`, Gradio) usa as mesmas classes que os scripts (`TissuePredictor` e `MoAPredictor`). Testei pela API: a aba de tecido deu cólon 0.32 e cerebelo 0.77 nas fotos, as mesmas respostas do script. Depois da refatoração, o `predict.py` deu as mesmas respostas de antes na semana 10.
