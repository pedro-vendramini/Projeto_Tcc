# 🧭 Menu Principal – Estrutura e Funções

## 1. 🧠 Treinar modelo
- **O que faz:** Treina um modelo Random Forest com base em amostras vetoriais (.gpkg) e raster (.tif)
- **Entrada necessária:**
  - Arquivo .gpkg com polígonos e classes
  - Raster .tif com bandas (RGB ou mais)
  - Nome do campo de classe
  - Número de árvores
- **Saída:** Arquivo .pkl do modelo

## 2. 🧮 Classificar raster (Threads - Modelos leves)
- **O que faz:** Classifica imagem raster usando modelo .pkl com threads (CPU leve)
- **Entrada:** Modelo .pkl, Raster .tif, Bloco, CPU (%)
- **Saída:** Raster classificado .tif + relatório .txt

## 3. 🧮 Classificar raster (Process - Modelos pesados)
- **Igual à anterior**, mas usa multiprocessing (mais eficiente para grandes modelos)

## 4. 🧮 Classificar raster em GRUPO
- **O que faz:** Classifica todos os rasters de uma pasta
- **Entrada:** Modelo .pkl + qualquer raster da pasta
- **Saída:** Todos classificados em nova pasta

## 5. 🧼 Limpar ruído (Filtro de Modo)
- **O que faz:** Suaviza ruído aplicando filtro de maioria
- **Entrada:** Raster .tif + tamanho da janela
- **Saída:** Novo raster com _modoN

## 6. 🧩 Segmentar rasters
- **O que faz:** Divide raster em blocos menores
- **Entrada:** Raster .tif + tamanho do bloco (px)
- **Saída:** Blocos .tif em pasta separada

## 7. 🧩 Unificar rasters
- **O que faz:** Junta blocos em um mosaico
- **Entrada:** Qualquer .tif da pasta + nome do mosaico
- **Saída:** mosaico.tif

## 8. 🔎 Analisar raster
- **O que faz:** Gera relatório com estatísticas por classe
- **Entrada:** Raster classificado .tif
- **Saída:** Relatório .txt com percentuais e áreas

## 9. 🖼️ Comparar rasters
- **O que faz:** Compara dois rasters pixel a pixel
- **Entrada:** Raster 1 + Raster 2
- **Saída:** Relatório .txt + raster de diferenças opcional

## 10. 🧹 Remover banda 4 (imagem RGB)
- **O que faz:** Remove banda 4 para obter imagem RGB pura
- **Entrada:** Raster com 4 bandas
- **Saída:** Raster com 3 bandas (RGB)

## 11. 🧹 Limpar prompt
- **O que faz:** Limpa a tela e reexibe banner

## 12. ❌ Sair
- Finaliza o programa

## Desenvolverdor:
- Pedro Aguiar Vendramini
- pedrovendramini.eng@gmail.com
