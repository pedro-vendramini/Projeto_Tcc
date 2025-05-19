# 🛰️ Classificador de Rasters com Random Forest – TCC UFRR

Este projeto tem como objetivo realizar a **classificação de uso e ocupação do solo** utilizando **modelos de aprendizado supervisionado** (Random Forest) aplicados a **imagens de satélite** no formato raster.

Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) no curso de **Engenharia Civil da Universidade Federal de Roraima – UFRR**.

## 📌 Funcionalidades principais

O script oferece uma interface interativa em terminal para as seguintes ações:

### 🧠 Treinamento de modelos
- Treinamento de modelos Random Forest com base em amostras vetoriais (formato GPKG).
- Extração automática de pixels e geração de modelos `.pkl`.

### 🧮 Classificação de imagens
- Classificação de uma única imagem ou em lote (pastas segmentadas).
- Suporte a paralelização com `ThreadPoolExecutor` e `ProcessPoolExecutor`.
- Geração de relatórios automáticos com tempo, parâmetros e pixels classificados.

### 🧩 Segmentação e unificação
- Segmentação de rasters em blocos de tamanho configurável.
- Segmentação com base em feições vetoriais.
- Unificação (mosaico) de rasters em um único arquivo.

### 🔎 Análises e verificações
- Estatísticas por classe (área, percentual, total de pixels).
- Comparação entre rasters (com percentual de igualdade).
- Verificação de resolução de imagens em lote.

### 📊 Matrizes de confusão
- Avaliação da acurácia por raster de referência.
- Avaliação com base em vetor de amostras.
- Geração de relatórios com métricas por classe.

### 🧼 Pós-processamento
- Filtro de modo para remoção de ruído em rasters classificados.
- Remoção da 4ª banda (alfa/transparência) de imagens RGB.

## 👨‍🎓 Sobre o autor
- Pedro Aguiar Vendramini
- Curso de Engenharia Civil – Universidade Federal de Roraima (UFRR)
- Email: pedrovendramini.eng@gmail.com

---

## 🛠️ Requisitos

### Bibliotecas principais:
- `numpy`, `pandas`
- `rasterio`, `geopandas`
- `scikit-learn`, `joblib`, `scipy`
- `questionary`, `tqdm`

Instale com:

```bash
pip install -r requirements.txt
