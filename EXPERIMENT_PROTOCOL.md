# Protocolo experimental CPU-only — RDD2022

## 1. Pergunta de pesquisa

Avaliar a qualidade e a variacao entre dominios de um detector compacto treinado
sob um orcamento de CPU explicito, com mudancas de pais, sensor, resolucao,
clima e aparencia da via.

## 2. Unidade experimental

- Imagem de estrada como unidade de inferencia.
- Quatro alvos oficiais: `D00`, `D10`, `D20` e `D40`.
- Classes adicionais permanecem documentadas, mas nao sao alvos do benchmark principal.
- Imagens sem uma classe oficial sao preservadas como negativas.

## 3. Limpeza congelada

1. Remover caixas com largura ou altura nao positiva.
2. Remover caixas fora dos limites da imagem; nao corrigir silenciosamente.
3. Converter coordenadas para COCO e YOLO a partir da mesma tabela canonica.
4. Preservar o teste original sem rotulos apenas para submissao ou inferencia final.
5. Versionar semente, manifesto, regras e checksums dos artefatos.

## 4. Splits

### Benchmark interno principal

- 80% treino, 10% validacao e 10% teste, criados somente sobre as 38.385 imagens originalmente rotuladas.
- Divisao feita separadamente em cada dominio.
- Identificadores consecutivos sao agrupados em blocos de 50 imagens.
- Um bloco nunca aparece em mais de um split, reduzindo vazamento por quadros visualmente proximos.
- O balanceamento considera quantidade de imagens, negativos e instancias de cada classe.

### Subconjunto de treino limitado por CPU

- Selecionar 2.800 imagens do split de treino, exatamente 400 por dominio.
- Dentro de cada dominio, preservar por amostragem estratificada a prevalencia
  de imagens negativas.
- Usar semente 2026 e registrar a lista absoluta, contagens realizadas por
  classe/dominio e o manifesto gerador.
- Manter validacao e teste completos; o limite aplica-se somente ao treino.
- Reportar cada dominio do teste separadamente e a media macro, sem chamar essa
  avaliacao de generalizacao para dominio nao visto.

### Teste original do desafio

As 9.035 imagens sem XML nao entram em selecao de hiperparametros nem na avaliacao interna. Usar apenas ao final, caso exista servidor de avaliacao ou protocolo externo comparavel.

## 5. Modelo e orcamento congelados

- YOLO11n inicializado com os pesos COCO oficiais.
- Entrada 320 x 320, batch 8, 10 epochs e 28.000 apresentacoes de imagem.
- AdamW, taxa inicial 0,00125, taxa final relativa 0,05, warm-up de um epoch.
- Flip horizontal, variacao HSV, escala/translacao e mosaic nos oito primeiros
  epochs; mosaic desligado nos dois finais.
- CPU Intel Core i5-10500T, oito threads PyTorch; nenhuma GPU ou precisao mista.
- Faster R-CNN MobileNetV3 e os pilotos curtos permanecem qualificacoes de
  engenharia e nao substituem o resultado no teste interno.

## 6. Metricas

- Primaria: COCO mAP de IoU 0,50 a 0,95.
- Secundarias: AP50, AP75, AP por classe e AP por dominio.
- Escala: AP para objetos pequenos, medios e grandes.
- Operacao: latencia, FPS, parametros, tempo de treino e memoria de pico no
  mesmo hardware.
- Incerteza: a execucao principal usa uma semente deterministica; a ausencia de
  repeticoes de treino deve ser declarada como limitacao, sem fabricar desvio
  padrao ou intervalo de variancia entre seeds.

## 7. Ablacoes prioritarias

1. Curva de qualificacao curta: 100 versus 500 passos no detector two-stage.
2. Faster R-CNN MobileNetV3 versus YOLO11n como decisao de custo de engenharia.
3. Analise de erros por classe, dominio, escala e imagem negativa.
4. Ablacoes sem negativos, tiling e resolucao maior ficam como trabalho futuro
   se nao forem executadas antes do congelamento.

## 8. Regras contra contaminacao

- Teste interno nunca orienta hiperparametros.
- Validacao escolhe checkpoint e hiperparametros; teste e executado apenas com a configuracao congelada.
- Blocos sequenciais e eventuais duplicatas perceptuais nao cruzam splits.
- Toda tabela informa semente, split, tamanho de entrada, hardware e versao do codigo.

## 9. Criterio de conclusao experimental

O conjunto principal esta completo quando houver: run CPU congelado, uma unica
inferencia no teste interno, metricas COCO agregadas e por dominio/classe,
analise de negativos, auditoria dos manifests, paper sem placeholders e PDF IEEE
compilado. Nenhum resultado de validacao pode ser apresentado como teste.

Status em 2026-09-03: criterio satisfeito. A avaliacao final esta em
`outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation/`, a proveniencia em
`paper/RESULTS_PROVENANCE.md`, e o manuscrito compilado em `paper/main.pdf`.
