print("Importando bibliotecas... Aguarde...")

# === Bibliotecas padrão ===
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# === Bibliotecas de terceiros ===
import questionary
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy.ndimage import generic_filter

# === Raster e geoprocessamento ===
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.merge import merge

# === Funções utilitárias ===

def exibir_banner():
    # Site:https://patorjk.com/software/taag/
    cores_ansi = {
        "preto": "\033[30m",
        "vermelho": "\033[31m",
        "verde": "\033[32m",
        "amarelo": "\033[33m",
        "azul": "\033[34m",
        "magenta": "\033[35m",
        "ciano": "\033[36m",
        "branco": "\033[37m",
        "verde_claro": "\033[92m",
        "azul_claro": "\033[94m",
        "reset": "\033[0m"
    }

    cor_banner = cores_ansi["ciano"]  # <<< ALTERAR AQUI A COR
    reset = cores_ansi["reset"]

    banner1 = r"""
    ████████╗ ██████╗ ██████╗
    ╚══██╔══╝██╔════╝██╔════╝
       ██║   ██║     ██║     
       ██║   ██║     ██║     
       ██║   ╚██████╗╚██████╗
       ╚═╝    ╚═════╝ ╚═════╝

    Projeto desenvolvido por: Pedro Aguiar Vendramini
    Contato: pedrovendramini.eng@gmail.com
    """

    banner2 = r"""
        -----------------------------------------------
            CLASSIFICADOR RANDOM FOREST DE RASTERS
            Universidade Federal de Roraima – UFRR
              Projeto para TCC - Engenharia Civil
                  Pedro Aguiar Vendramini
                pedrovendramini.eng@gmail.com
        -----------------------------------------------
    """

    banner = banner2
    print(cor_banner + banner + reset)
 
def selecionar_arquivo_com_extensoes(extensoes, pasta_inicial=".", mensagem="Selecione um arquivo:"):
    pasta_atual = os.path.abspath(pasta_inicial)

    while True:
        try:
            itens = os.listdir(pasta_atual)
        except PermissionError:
            print(f"Sem permissão para acessar: {pasta_atual}")
            pasta_atual = os.path.dirname(pasta_atual)
            continue

        arquivos = [
            f for f in sorted(itens)
            if os.path.isfile(os.path.join(pasta_atual, f)) and any(f.lower().endswith(ext) for ext in extensoes)
        ]
        diretorios = [
            f for f in sorted(itens)
            if os.path.isdir(os.path.join(pasta_atual, f))
        ]

        opcoes = []
        if os.path.dirname(pasta_atual) != pasta_atual:
            opcoes.append("↩️ Voltar")
        opcoes += arquivos
        opcoes += [f"[PASTA] {f}" for f in diretorios]

        if not opcoes:
            print(f"Nenhum arquivo ou pasta relevante em: {pasta_atual}")
            return None

        titulo = f"{mensagem} (📁 {os.path.basename(pasta_atual) or '/'})"
        escolha = questionary.select(titulo, choices=opcoes).ask()

        if escolha is None:
            print("\n🔄 Operação cancelada. Reiniciando...\n")
            python = sys.executable
            os.execl(python, python, *sys.argv)

        if escolha == "↩️ Voltar":
            pasta_atual = os.path.dirname(pasta_atual)
        elif escolha.startswith("[PASTA] "):
            subpasta = escolha.replace("[PASTA] ", "")
            pasta_atual = os.path.join(pasta_atual, subpasta)
        else:
            return os.path.join(pasta_atual, escolha)
    
def listar_arquivos(extensoes):
    arquivos = [f for f in os.listdir('.') if os.path.isfile(f) and any(f.lower().endswith(ext) for ext in extensoes)]
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado com as extensões: {', '.join(extensoes)}")
    return arquivos

def Limpar():
    os.system('cls' if os.name == 'nt' else 'clear')
    exibir_banner()

# === Funções do programa ===
def treinar_modelo():
    vetor_amostras = selecionar_arquivo_com_extensoes([".gpkg"], mensagem="Selecione o arquivo de amostras (GPKG):")
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster de entrada (TIF):")

    nome_raster = os.path.splitext(os.path.basename(raster_path))[0]
    campo_classe = questionary.text("Nome do campo com o valor da classe:").ask()
    n_arvores = int(questionary.text("Número de árvores no modelo Random Forest:").ask())

    nome_sugerido = f"{nome_raster}-N{n_arvores}-modelo"
    modelo_saida = questionary.text("Nome do arquivo de saída para o modelo (sem extensão):", default=nome_sugerido).ask()

    if not modelo_saida.lower().endswith(".pkl"):
        modelo_saida += ".pkl"

    print("[1] Iniciando carregamento das amostras vetoriais...")
    gdf = gpd.read_file(vetor_amostras)
    gdf = gdf[[campo_classe, "geometry"]].dropna(subset=["geometry"])
    print(f"[1.1] Total de amostras (polígonos) carregadas: {len(gdf)}")

    features, labels = [], []
    total_pixels = 0

    print("[2] Processando polígonos e extraindo pixels do raster...")
    with rasterio.open(raster_path) as src:
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extraindo pixels"):
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = mask(src, geom, crop=True)
            except:
                continue
            data = out_image.reshape(out_image.shape[0], -1).T
            data = data[~np.any(data == src.nodata, axis=1)]
            if data.shape[0] == 0:
                continue
            features.append(data)
            labels.extend([row[campo_classe]] * data.shape[0])
            total_pixels += data.shape[0]

    print(f"[2.1] Total de pixels extraídos para treinamento: {total_pixels}")

    X = np.vstack(features)
    y = np.array(labels)

    if X.shape[0] > 10_000_000:
        print(f"[2.2] Reduzindo para 10.000.000 de pixels (amostragem aleatória com balanceamento)...")
        X, y = resample(X, y, n_samples=10_000_000, random_state=42)

    print("[3] Treinando modelo Random Forest...")
    clf = RandomForestClassifier(n_estimators=n_arvores, n_jobs=-1)
    clf.fit(X, y)
    dump(clf, modelo_saida)
    print(f"[✔] Modelo salvo: {modelo_saida}")

def classificar_imagem_thread():
    modelo_path = selecionar_arquivo_com_extensoes([".pkl"], mensagem="Selecione o modelo .pkl treinado:")
    raster_entrada = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster a ser classificado (TIF):")

    total_cores = os.cpu_count() or 1
    if total_cores >= 8:
        bloco_padrao = "2048"
    elif total_cores >= 4:
        bloco_padrao = "1024"
    else:
        bloco_padrao = "512"

    print("\n📌 O 'tamanho do bloco' define o pedaço da imagem que será processado por vez.")
    print("Valores maiores aceleram o processo, mas exigem mais memória.")
    print("Valores menores são mais seguros, mas mais lentos.\n")
    tamanho_bloco = int(questionary.select(
        "Escolha o tamanho dos blocos (em pixels):",
        choices=["512", "1024", "2048", "4096"],
        default=bloco_padrao
    ).ask())

    if total_cores <= 2:
        cpu_padrao = "100"
    elif total_cores <= 4:
        cpu_padrao = "85"
    else:
        cpu_padrao = "60"

    print("\n⚙️  O uso de CPU define quantas threads serão usadas para paralelizar o processamento.")
    print("Valores altos aceleram, mas consomem mais recursos (pode aquecer o PC ou travar outras tarefas).")
    print("💡 Recomendado: 60% para uso geral. Use 100% apenas se não estiver usando o computador para mais nada.\n")

    uso_cpu_percentual = int(questionary.text(f"Quantos % da CPU deseja utilizar? (ex: {cpu_padrao})", default=cpu_padrao).ask())
    uso_cpu_percentual = max(1, min(100, uso_cpu_percentual))

    n_threads = max(1, int((uso_cpu_percentual / 100) * total_cores))
    nome_base, extensao = os.path.splitext(raster_entrada)
    raster_saida_base = f"{nome_base}-classificado"
    raster_saida = raster_saida_base + extensao
    contador = 1
  
    while os.path.exists(raster_saida):
        raster_saida = f"{raster_saida_base}-v{contador}{extensao}"
        contador += 1
    relatorio_saida = raster_saida.replace(extensao, "-relatorio.txt")

    print(f"\n[1] Carregando modelo com {n_threads} thread(s)...")
    modelo = load(modelo_path)

    print("[2] Abrindo imagem raster...")
    start_time = time.time()
    with rasterio.open(raster_entrada) as src:
        profile = src.profile.copy()
        profile.update(dtype='uint8', count=1, compress='lzw')
        largura, altura, bandas = src.width, src.height, src.count
        nodata = src.nodata
        total_pixels = largura * altura
        print(f"[2.1] Total de pixels a serem classificados: {total_pixels}")

        tarefas = [(Window(j, i, min(tamanho_bloco, largura - j), min(tamanho_bloco, altura - i)),
                     src.read(window=Window(j, i, min(tamanho_bloco, largura - j), min(tamanho_bloco, altura - i)), boundless=True),
                     nodata)
                   for i in range(0, altura, tamanho_bloco)
                   for j in range(0, largura, tamanho_bloco)]

        def classificar_bloco(args):
            window, bloco, nodata = args
            bloco = bloco.transpose(1, 2, 0)
            mascara_valida = ~np.any(bloco == nodata, axis=2)
            matriz_saida = np.zeros((bloco.shape[0], bloco.shape[1]), dtype='uint8')
            if np.any(mascara_valida):
                bloco_2d = bloco[mascara_valida]
                previsoes = modelo.predict(bloco_2d)
                matriz_saida[mascara_valida] = previsoes.astype('uint8')
            return (window, matriz_saida)

        with rasterio.open(raster_saida, 'w', **profile) as dst:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(classificar_bloco, tarefa) for tarefa in tarefas]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Classificando"):
                    window, matriz_saida = future.result()
                    dst.write(matriz_saida, 1, window=window)

    tempo_total = round(time.time() - start_time, 2)
    print(f"[✔] Classificação salva como: {raster_saida}")
    print(f"[⏱] Tempo total decorrido: {tempo_total} segundos")

    with open(relatorio_saida, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO DE IMAGEM\n")
        f.write(f"Data e hora: {datetime.now()}\n\n")
        f.write(f"Modelo utilizado: {modelo_path}\n")
        f.write(f"Raster de entrada: {raster_entrada}\n")
        f.write(f"Raster de saída: {raster_saida}\n\n")
        f.write(f"Tamanho do bloco: {tamanho_bloco} pixels\n")
        f.write(f"Uso de CPU: {uso_cpu_percentual}% ({n_threads} thread(s))\n")
        f.write(f"Total de pixels classificados: {total_pixels}\n")
        f.write(f"Tempo total de execução: {tempo_total} segundos\n")
      
    print(f"[📝] Relatório salvo como: {relatorio_saida}")

def classificar_bloco_serializado(args):
    window, bloco, nodata, modelo_bytes = args
    import pickle
    modelo = pickle.loads(modelo_bytes)
    bloco = bloco.transpose(1, 2, 0)
    mascara_valida = ~np.any(bloco == nodata, axis=2)
    matriz_saida = np.zeros((bloco.shape[0], bloco.shape[1]), dtype='uint8')
    if np.any(mascara_valida):
        bloco_2d = bloco[mascara_valida]
        previsoes = modelo.predict(bloco_2d)
        matriz_saida[mascara_valida] = previsoes.astype('uint8')
    return (window, matriz_saida)

def classificar_imagem_pool():
    modelo_path = selecionar_arquivo_com_extensoes([".pkl"], mensagem="Selecione o modelo .pkl treinado:")
    raster_entrada = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster a ser classificado (TIF):")

    total_cores = os.cpu_count() or 1
    bloco_padrao = "2048" if total_cores >= 8 else "1024" if total_cores >= 4 else "512"

    print("\n📌 O 'tamanho do bloco' define o pedaço da imagem que será processado por vez.")
    print("Valores maiores aceleram o processo, mas exigem mais memória.")
    print("Valores menores são mais seguros, mas mais lentos.\n")
    tamanho_bloco = int(questionary.select(
        "Escolha o tamanho dos blocos (em pixels):",
        choices=["512", "1024", "2048", "4096"],
        default=bloco_padrao
    ).ask())

    cpu_padrao = "100" if total_cores <= 2 else "85" if total_cores <= 4 else "60"

    print("\n⚙️  O uso de CPU define quantas threads serão usadas para paralelizar o processamento.")
    print("Valores altos aceleram, mas consomem mais recursos (pode aquecer o PC ou travar outras tarefas).")
    print("💡 Recomendado: 60% para uso geral. Use 100% apenas se não estiver usando o computador para mais nada.\n")

    uso_cpu_percentual = int(questionary.text(f"Quantos % da CPU deseja utilizar? (ex: {cpu_padrao})", default=cpu_padrao).ask())
    uso_cpu_percentual = max(1, min(100, uso_cpu_percentual))

    n_threads = max(1, int((uso_cpu_percentual / 100) * total_cores))
    nome_base, extensao = os.path.splitext(raster_entrada)
    raster_saida_base = f"{nome_base}-classificado"
    raster_saida = raster_saida_base + extensao
    contador = 1
    while os.path.exists(raster_saida):
        raster_saida = f"{raster_saida_base}-v{contador}{extensao}"
        contador += 1
    relatorio_saida = raster_saida.replace(extensao, "-relatorio.txt")

    print(f"\n[1] Carregando modelo com {n_threads} processo(s)...")
    import pickle
    modelo = load(modelo_path)
    modelo_bytes = pickle.dumps(modelo)

    print("[2] Abrindo imagem raster...")
    start_time = time.time()
    with rasterio.open(raster_entrada) as src:
        profile = src.profile.copy()
        profile.update(dtype='uint8', count=1, compress='lzw')
        largura, altura, bandas = src.width, src.height, src.count
        nodata = src.nodata
        total_pixels = largura * altura
        print(f"[2.1] Total de pixels a serem classificados: {total_pixels}")

        tarefas = [(Window(j, i, min(tamanho_bloco, largura - j), min(tamanho_bloco, altura - i)),
                     src.read(window=Window(j, i, min(tamanho_bloco, largura - j), min(tamanho_bloco, altura - i)), boundless=True),
                     nodata,
                     modelo_bytes)
                   for i in range(0, altura, tamanho_bloco)
                   for j in range(0, largura, tamanho_bloco)]

        with rasterio.open(raster_saida, 'w', **profile) as dst:
            with ProcessPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(classificar_bloco_serializado, tarefa) for tarefa in tarefas]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Classificando"):
                    window, matriz_saida = future.result()
                    dst.write(matriz_saida, 1, window=window)

    tempo_total = round(time.time() - start_time, 2)
    print(f"[✔] Classificação salva como: {raster_saida}")
    print(f"[⏱] Tempo total decorrido: {tempo_total} segundos")

    with open(relatorio_saida, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO DE IMAGEM\n")
        f.write(f"Data e hora: {datetime.now()}\n\n")
        f.write(f"Modelo utilizado: {modelo_path}\n")
        f.write(f"Raster de entrada: {raster_entrada}\n")
        f.write(f"Raster de saída: {raster_saida}\n\n")
        f.write(f"Tamanho do bloco: {tamanho_bloco} pixels\n")
        f.write(f"Uso de CPU: {uso_cpu_percentual}% ({n_threads} processo(s))\n")
        f.write(f"Total de pixels classificados: {total_pixels}\n")
        f.write(f"Tempo total de execução: {tempo_total} segundos\n")

    print(f"[📝] Relatório salvo como: {relatorio_saida}")

def classificar_rasters_segmentados():
    modelo_path = selecionar_arquivo_com_extensoes([".pkl"], mensagem="Selecione o modelo .pkl treinado:")
    raster_exemplo = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione um dos rasters segmentados para definir a pasta:")
    pasta_segmentos = os.path.dirname(raster_exemplo)
    arquivos_tif = [os.path.join(pasta_segmentos, f) for f in sorted(os.listdir(pasta_segmentos)) if f.lower().endswith(".tif")]

    nome_padrao = os.path.basename(pasta_segmentos.rstrip("/\\")) + "_classificados"
    nome_final = questionary.text("Nome da pasta para salvar os rasters classificados:", default=nome_padrao).ask()
    pasta_saida = os.path.join(os.path.dirname(pasta_segmentos), nome_final)
    os.makedirs(pasta_saida, exist_ok=True)

    print(f"[📂] Classificando {len(arquivos_tif)} rasters na pasta: {pasta_segmentos}")
    print(f"[🧠] Usando modelo: {modelo_path}")

    modelo = load(modelo_path)

    for raster_entrada in tqdm(arquivos_tif, desc="Classificando imagens"):
        nome_base, extensao = os.path.splitext(os.path.basename(raster_entrada))
        raster_saida = os.path.join(pasta_saida, f"{nome_base}-classificado{extensao}")

        # Verificar se a classificação já foi feita e o raster está completo
        if os.path.exists(raster_saida):
            try:
                with rasterio.open(raster_entrada) as src_in, rasterio.open(raster_saida) as src_out:
                    if src_in.shape == src_out.shape:
                        print(f"[⏩] Pulando '{nome_base}' (já classificado e completo)")
                        continue
            except:
                pass  # Se não puder abrir, tenta classificar de novo

        with rasterio.open(raster_entrada) as src:
            profile = src.profile.copy()
            profile.update(dtype='uint8', count=1, compress='lzw')
            largura, altura = src.width, src.height
            nodata = src.nodata

            bloco = src.read()
            bloco = bloco.transpose(1, 2, 0)
            mascara_valida = ~np.any(bloco == nodata, axis=2)
            matriz_saida = np.zeros((altura, largura), dtype='uint8')
            if np.any(mascara_valida):
                bloco_2d = bloco[mascara_valida]
                previsoes = modelo.predict(bloco_2d)
                matriz_saida[mascara_valida] = previsoes.astype('uint8')

            with rasterio.open(raster_saida, 'w', **profile) as dst:
                dst.write(matriz_saida, 1)

    print(f"[✔] Classificação concluída. Resultados salvos em: {pasta_saida}")

def remover_banda_4():
    nome_entrada = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o arquivo TIFF (ex: imagem.tif):")
    entrada = os.path.abspath(nome_entrada)
    nome_base, extensao = os.path.splitext(nome_entrada)
    saida = f"{nome_base}_RGB{extensao}"

    with rasterio.open(entrada) as src:
        if src.count < 3:
            raise ValueError("A imagem não possui ao menos 3 bandas (RGB).")
        profile = src.profile
        profile.update(count=3)
        with rasterio.open(saida, 'w', **profile) as dst:
            for i in range(1, 4):
                dst.write(src.read(i), i)
    print(f"[✔] Imagem salva sem banda alfa como: {saida}")

def comparar_rasters():
    raster1 = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o primeiro raster:")
    raster2 = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o segundo raster:")
    pasta_script = os.path.dirname(os.path.abspath(__file__))

    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        if src1.shape != src2.shape or src1.transform != src2.transform:
            raise ValueError("Os rasters não têm a mesma dimensão ou transformação espacial.")

        array1 = src1.read(1)
        array2 = src2.read(1)
        mask_valid = np.ones(array1.shape, dtype=bool)
        if src1.nodata is not None:
            mask_valid &= array1 != src1.nodata
        if src2.nodata is not None:
            mask_valid &= array2 != src2.nodata

        iguais = (array1 == array2) & mask_valid
        diferentes = (~iguais) & mask_valid
        total_validos = np.sum(mask_valid)
        total_iguais = np.sum(iguais)
        total_diferentes = total_validos - total_iguais
        percentual = (total_iguais / total_validos) * 100 if total_validos > 0 else 0.0

        data_str = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

        print("\n📊 RESULTADO DA COMPARAÇÃO")
        print(f"Raster 1: {raster1}")
        print(f"Raster 2: {raster2}")
        print(f"Total de pixels válidos: {total_validos}")
        print(f"Pixels iguais: {total_iguais}")
        print(f"Pixels diferentes: {total_diferentes}")
        print(f"Percentual de igualdade: {percentual:.2f}%")

        salvar_relatorio = questionary.select(
            "Deseja salvar o relatório da comparação?",
            choices=["Sim", "Não"],
            default="Sim"
        ).ask() == "Sim"

        salvar_raster_diferencas = questionary.select(
            "Deseja gerar o raster de diferenças?",
            choices=["Sim", "Não"],
            default="Sim"
        ).ask() == "Sim"

        if salvar_relatorio:
            relatorio_path = os.path.join(pasta_script, f"relatorio_comparacao_{data_str}.txt")
            with open(relatorio_path, "w", encoding="utf-8") as f:
                f.write("RELATÓRIO DE COMPARAÇÃO DE RASTERS\n")
                f.write(f"Data e hora: {datetime.now()}\n")
                f.write(f"Raster 1: {raster1}\n")
                f.write(f"Raster 2: {raster2}\n")
                f.write(f"Total válidos: {total_validos}\n")
                f.write(f"Pixels iguais: {total_iguais}\n")
                f.write(f"Pixels diferentes: {total_diferentes}\n")
                f.write(f"Percentual de igualdade: {percentual:.2f}%\n")
            print(f"[📝] Relatório salvo em: {relatorio_path}")

        if salvar_raster_diferencas:
            dif_array = np.full(array1.shape, 255, dtype=np.uint8)
            dif_array[iguais] = 0
            dif_array[diferentes] = 1
            profile = src1.profile
            profile.update(dtype=rasterio.uint8, count=1, nodata=255)

            diferencas_path = os.path.join(pasta_script, f"diferencas_{data_str}.tif")
            with rasterio.open(diferencas_path, "w", **profile) as dst:
                dst.write(dif_array, 1)
            print(f"[🗺️] Raster de diferenças salvo como: {diferencas_path}")

        print("\n[✔] Comparação concluída.")

def segmentar_raster_em_blocos():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster a ser segmentado:")
    bloco_pixels = int(questionary.text("Tamanho dos blocos (em pixels, ex: 1000):", default="1000").ask())

    nome_base = os.path.splitext(os.path.basename(raster_path))[0]
    pasta_saida = os.path.join(os.path.dirname(raster_path), f"{nome_base}_segmentos")

    os.makedirs(pasta_saida, exist_ok=True)

    with rasterio.open(raster_path) as src:
        largura, altura = src.width, src.height
        profile = src.profile.copy()

        count = 0
        for i in range(0, altura, bloco_pixels):
            for j in range(0, largura, bloco_pixels):
                h = min(bloco_pixels, altura - i)
                w = min(bloco_pixels, largura - j)
                window = Window(j, i, w, h)
                transform = src.window_transform(window)
                bloco = src.read(window=window)

                profile.update({
                    "height": h,
                    "width": w,
                    "transform": transform
                })

                nome_bloco = os.path.join(pasta_saida, f"{nome_base}_bloco_{i}_{j}.tif")
                with rasterio.open(nome_bloco, "w", **profile) as dst:
                    dst.write(bloco)
                count += 1

    print(f"[✔] Segmentação concluída. Total de blocos salvos: {count}")
    print(f"[📁] Arquivos salvos em: {pasta_saida}")

def unir_rasters_em_mosaico():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione qualquer raster da pasta a ser unida:")
    if not raster_path:
        return

    pasta_base = os.path.dirname(raster_path)
    arquivos_tif = [os.path.join(pasta_base, f) for f in sorted(os.listdir(pasta_base)) if f.lower().endswith(".tif")]

    if not arquivos_tif:
        print("[⚠] Nenhum arquivo .tif encontrado na pasta.")
        return

    nome_saida = questionary.text("Nome desejado para o arquivo mosaico (sem .tif):", default="mosaico").ask()
    saida_path = os.path.join(pasta_base, f"{nome_saida}.tif")

    print(f"[🔄] Unindo {len(arquivos_tif)} rasters da pasta '{os.path.basename(pasta_base)}'...")

    fontes = [rasterio.open(fp) for fp in arquivos_tif]
    mosaico, transformacao = merge(fontes)
    perfil = fontes[0].profile

    perfil.update({
        "height": mosaico.shape[1],
        "width": mosaico.shape[2],
        "transform": transformacao,
        "compress": "lzw"
    })

    with rasterio.open(saida_path, "w", **perfil) as dst:
        dst.write(mosaico)

    print(f"[✔] Mosaico salvo como: {saida_path}")

def analisar_raster():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster classificado para análise estatística:")

    with rasterio.open(raster_path) as src:
        array = src.read(1)
        transform = src.transform
        nodata = src.nodata

        if nodata is not None:
            array = array[array != nodata]

        if array.size == 0:
            print("[⚠] O raster não contém dados válidos.")
            return

        classes, counts = np.unique(array, return_counts=True)
        total = counts.sum()
        pixel_area_m2 = abs(transform[0] * transform[4])  # geralmente: 0.3 * -0.3 = 0.09 m²

        nome_base = os.path.splitext(os.path.basename(raster_path))[0]
        pasta = os.path.dirname(raster_path)
        relatorio_path = os.path.join(pasta, f"{nome_base}_relatorio_estatistico.txt")

        with open(relatorio_path, "w", encoding="utf-8") as rel:
            rel.write("RELATÓRIO ESTATÍSTICO POR CLASSE\n")
            rel.write(f"Raster analisado: {raster_path}\n")
            rel.write(f"Data: {datetime.now()}\n")
            rel.write(f"Tamanho do pixel: {pixel_area_m2:.4f} m²\n")
            rel.write(f"Total de pixels válidos: {total}\n\n")
            rel.write("CLASSE | PIXELS | PERCENTUAL | ÁREA (m²) | ÁREA (ha)\n")
            rel.write("------------------------------------------------------------\n")
            for cls, count in zip(classes, counts):
                percentual = (count / total) * 100
                area_m2 = count * pixel_area_m2
                area_ha = area_m2 / 10_000
                rel.write(f"{cls:^6} | {count:^7} | {percentual:9.2f}% | {area_m2:10.2f} | {area_ha:9.2f}\n")

        print(f"[📊] Relatório gerado: {relatorio_path}")

def modo_local(pixels, nodata):
    valores = pixels[pixels != nodata] if nodata is not None else pixels
    if len(valores) == 0:
        return nodata if nodata is not None else 0
    return np.bincount(valores.astype(int)).argmax()

def processar_bloco_vertical(args):
    array, start, end, nodata, tamanho_janela = args
    bloco = array[start:end, :]
    filtrado = generic_filter(
        bloco,
        lambda p: modo_local(p, nodata),
        size=tamanho_janela,
        mode='nearest'
    )
    return (start, filtrado)

def aplicar_filtro_modo():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster classificado para aplicar o filtro de modo:")
    tamanho_janela = int(questionary.text("Tamanho da janela (ímpar, ex: 3, 5, 7):", default="3").ask())

    if tamanho_janela % 2 == 0:
        print("[⚠] O tamanho da janela deve ser um número ímpar.")
        return

    with rasterio.open(raster_path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    altura, largura = array.shape

    total_cores = mp.cpu_count()
    cpu_padrao = "100" if total_cores <= 2 else "85" if total_cores <= 4 else "60"

    print("\n⚙️  Defina o uso da CPU para paralelismo do filtro de modo")
    uso_cpu_percentual = int(questionary.text(f"Quantos % da CPU deseja utilizar? (ex: {cpu_padrao})", default=cpu_padrao).ask())
    uso_cpu_percentual = max(1, min(100, uso_cpu_percentual))
    n_processos = max(1, int((uso_cpu_percentual / 100) * total_cores))

    print(f"[🔄] Aplicando filtro de modo 2D por blocos com {n_processos} processo(s)...")

    bloco_altura = 256  # Ajustável: número de linhas por bloco
    blocos = []
    for i in range(0, altura, bloco_altura):
        inicio = i
        fim = min(i + bloco_altura, altura)
        blocos.append((array, inicio, fim, nodata, tamanho_janela))

    array_filtrado = np.zeros_like(array)

    with mp.Pool(processes=n_processos) as pool:
        for inicio, resultado in tqdm(pool.imap(processar_bloco_vertical, blocos), total=len(blocos), desc="Filtrando blocos"):
            array_filtrado[inicio:inicio+resultado.shape[0], :] = resultado

    nome_base, extensao = os.path.splitext(os.path.basename(raster_path))
    saida_path = os.path.join(os.path.dirname(raster_path), f"{nome_base}_modo{tamanho_janela}{extensao}")

    with rasterio.open(saida_path, "w", **profile) as dst:
        dst.write(array_filtrado, 1)

    print(f"[✔] Filtro de modo aplicado. Raster salvo como: {saida_path}")

# === Menu principal ===
def menu():
    exibir_banner()
    Limpar()
    while True:
        print("\n")
        opcao = questionary.select(
            "Escolha uma ação:",
            choices=[
                "🧠 Treinar modelo",
                "🧮 Classificar raster (Threads - Modelos leves)",
                "🧮 Classificar raster (Process - Modelos pesados)",
                "🧮 Classificar raster em GRUPO",
                "🧼 Limpar ruído",
                "🧩 Segmentar rasters",
                "🧩 Unificar rasters",
                "🔎 Analisar raster",
                "🖼️ Comparar rasters",
                "🧹 Remover banda 4 (imagem RGB)",
                "🧹 Limpar prompt",
                "❌ Sair"]
        ).ask()

        if opcao == "🧠 Treinar modelo":
            treinar_modelo()
        elif opcao == "🧮 Classificar raster (Threads - Modelos leves)":
            classificar_imagem_thread()
        elif opcao == "🧮 Classificar raster (Process - Modelos pesados)":
            classificar_imagem_pool()
        elif opcao == "🧮 Classificar raster em GRUPO":
            classificar_rasters_segmentados()
        elif opcao == "🧼 Limpar ruído":
            aplicar_filtro_modo()
        elif opcao == "🧩 Segmentar rasters":
            segmentar_raster_em_blocos()
        elif opcao == "🧩 Unificar rasters":
            unir_rasters_em_mosaico()
        elif opcao == "🔎 Analisar raster":
            analisar_raster()
        elif opcao == "🖼️ Comparar rasters":
            comparar_rasters()
        elif opcao == "🧹 Remover banda 4 (imagem RGB)":
            remover_banda_4()
        elif opcao == "🧹 Limpar prompt":
            Limpar()
        elif opcao == "❌ Sair":
            os.system('cls' if os.name == 'nt' else 'clear')
            break

if __name__ == "__main__":
    menu()
