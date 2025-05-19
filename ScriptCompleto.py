# === Bibliotecas padrão ===
import os
import sys
import time
import platform
import pickle
import multiprocessing as mp
import winsound
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# === Bibliotecas de terceiros ===
import numpy as np
import pandas as pd
import questionary
from joblib import dump, load
from questionary import Style
from scipy.ndimage import generic_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from tqdm import tqdm

# === Bibliotecas específicas de geoprocessamento ===
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.windows import Window

# === Configurações de estilo ===
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

estilo_personalizado_selecao = Style([
    ("pointer", "bold fg:yellow"),
    ("selected", "bold fg:white bg:blue"),
    ("highlighted", "fg:yellow"),
    ("answer", "bold fg:green"),
])

# === Funções utilitárias ===
def exibir_banner():
    # Site:https://patorjk.com/software/taag/

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
    
    banner3 = r"""
    ╔════════════════════════════════════════════╗
    ║   CLASSIFICAÇÃO DE USO E OCUPAÇÃO DO SOLO  ║
    ║     Modelo Random Forest aplicado via      ║
    ║             Python + Rasterio              ║
    ╠════════════════════════════════════════════╣
    ║ Universidade Federal de Roraima – UFRR     ║
    ║ Engenharia Civil – TCC                     ║
    ║ Autor: Pedro Aguiar Vendramini             ║
    ║ Contato: pedrovendramini.eng@gmail.com     ║
    ╚════════════════════════════════════════════╝
    """

    banner4 = r"""
    ===========================================================
    CLASSIFICAÇÃO DE RASTERS - RANDOM FOREST - TCC UFRR
    ===========================================================
    Autor : Pedro Aguiar Vendramini
    Curso : Engenharia Civil - Universidade Federal de Roraima
    Email : pedrovendramini.eng@gmail.com
    ===========================================================
    """

    banner = banner3
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
        escolha = questionary.select(titulo, choices=opcoes, style=estilo_personalizado_selecao).ask()

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

def alerta_conclusao(som=True):
    if som:
        sistema = platform.system()
        if sistema == "Windows":
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        else:
            # Para macOS e Linux: usa terminal para tocar um som padrão
            print('\a')  # Beep via terminal (pode não funcionar em todos os terminais)

def Limpar(banner=True):
    os.system('cls' if os.name == 'nt' else 'clear')
    if banner:
        exibir_banner()

### Funções do programa ###

## Treinar modelo

def treinar_modelo():
    vetor_amostras = selecionar_arquivo_com_extensoes([".gpkg"], mensagem="Selecione o arquivo de amostras (GPKG):")
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster de entrada (TIF):")

    print("[1] Lendo atributos disponíveis no vetor...")
    gdf_temp = gpd.read_file(vetor_amostras)
    campos_disponiveis = [col for col in gdf_temp.columns if col != "geometry"]
    campo_classe = questionary.select("Selecione o campo com o valor da classe:", choices=campos_disponiveis).ask()

    nome_vetor = os.path.splitext(os.path.basename(vetor_amostras))[0]
    n_arvores = int(questionary.text("Número de árvores no modelo Random Forest:").ask())

    nome_sugerido = f"{nome_vetor}-N{n_arvores}-modelo"
    modelo_saida_nome = questionary.text("Nome do arquivo de saída para o modelo (sem extensão):", default=nome_sugerido).ask()
    if not modelo_saida_nome.lower().endswith(".pkl"):
        modelo_saida_nome += ".pkl"

    pasta_padrao = os.path.join(os.path.dirname(__file__), "Modelos treinados")
    os.makedirs(pasta_padrao, exist_ok=True)

    salvar_em_padrao = questionary.confirm(f"Deseja salvar o modelo na pasta padrão? ({pasta_padrao})", default=True).ask()
    if salvar_em_padrao:
        caminho_modelo = os.path.join(pasta_padrao, modelo_saida_nome)
    else:
        pasta_customizada = questionary.path("Selecione a pasta onde deseja salvar o modelo:").ask()
        os.makedirs(pasta_customizada, exist_ok=True)
        caminho_modelo = os.path.join(pasta_customizada, modelo_saida_nome)

    print("[2] Iniciando carregamento das amostras vetoriais...")
    gdf = gdf_temp[[campo_classe, "geometry"]].dropna(subset=["geometry"])
    print(f"[2.1] Total de amostras (polígonos) carregadas: {len(gdf)}")

    features, labels = [], []
    total_pixels = 0

    print("[3] Processando polígonos e extraindo pixels do raster...")
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

    print(f"[3.1] Total de pixels extraídos para treinamento: {total_pixels}")

    X = np.vstack(features)
    y = np.array(labels)

    if X.shape[0] > 10_000_000:
        print("[3.2] Reduzindo para 10.000.000 de pixels (amostragem aleatória com balanceamento)...")
        X, y = resample(X, y, n_samples=10_000_000, random_state=42)

    print("[4] Treinando modelo Random Forest...")
    clf = RandomForestClassifier(n_estimators=n_arvores, n_jobs=-1)
    clf.fit(X, y)
    dump(clf, caminho_modelo)
    alerta_conclusao()
    print(f"[✔] Modelo salvo com sucesso: {caminho_modelo}")

## Calssificação de imagens

def configurar_classificacao_individual():
    modelo_path = selecionar_arquivo_com_extensoes([".pkl"], mensagem="Selecione o modelo .pkl treinado:")
    raster_entrada = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster a ser classificado (TIF):")

    total_cores = os.cpu_count() or 1
    bloco_padrao = "2048" if total_cores >= 8 else "1024" if total_cores >= 4 else "512"
    cpu_padrao = "100" if total_cores <= 2 else "85" if total_cores <= 4 else "60"

    print("\n📌 O 'tamanho do bloco' define o pedaço da imagem que será processado por vez.")
    tamanho_bloco = int(questionary.select(
        "Escolha o tamanho dos blocos (em pixels):",
        choices=["512", "1024", "2048", "4096"],
        default=bloco_padrao,
        style=estilo_personalizado_selecao
    ).ask())

    print("\n⚙️  O uso de CPU define quantas threads/processos serão usados para paralelizar o processamento.")
    uso_cpu_percentual = int(questionary.text(
        f"Quantos % da CPU deseja utilizar? (ex: {cpu_padrao})", default=cpu_padrao
    ).ask())
    uso_cpu_percentual = max(1, min(100, uso_cpu_percentual))
    n_threads = max(1, int((uso_cpu_percentual / 100) * total_cores))

    return modelo_path, raster_entrada, tamanho_bloco, uso_cpu_percentual, n_threads

def classificar_imagem_thread(modelo_path, raster_entrada, tamanho_bloco, uso_cpu_percentual, n_threads):
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
            mascara_valida = ~(np.all(bloco == 0, axis=2))
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

    alerta_conclusao()
    print(f"[📝] Relatório salvo como: {relatorio_saida}")

def classificar_imagem_pool(modelo_path, raster_entrada, tamanho_bloco, uso_cpu_percentual, n_threads):
    def classificar_bloco_serializado(args):
        window, bloco, nodata, modelo_bytes = args
        modelo = pickle.loads(modelo_bytes)
        bloco = bloco.transpose(1, 2, 0)
        mascara_valida = ~(np.all(bloco == 0, axis=2))
        matriz_saida = np.zeros((bloco.shape[0], bloco.shape[1]), dtype='uint8')
        if np.any(mascara_valida):
            bloco_2d = bloco[mascara_valida]
            previsoes = modelo.predict(bloco_2d)
            matriz_saida[mascara_valida] = previsoes.astype('uint8')
        return (window, matriz_saida)

    nome_base, extensao = os.path.splitext(raster_entrada)
    raster_saida_base = f"{nome_base}-classificado"
    raster_saida = raster_saida_base + extensao
    contador = 1

    while os.path.exists(raster_saida):
        raster_saida = f"{raster_saida_base}-v{contador}{extensao}"
        contador += 1
    relatorio_saida = raster_saida.replace(extensao, "-relatorio.txt")

    print(f"\n[1] Carregando modelo com {n_threads} processo(s)...")
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

    alerta_conclusao()
    print(f"[📝] Relatório salvo como: {relatorio_saida}")

def classificar_imagem():
    tipo = questionary.select(
        "Deseja classificar uma única imagem ou em grupo?",
        choices=["📄 Classificar uma imagem", "📁 Classificar várias imagens (grupo)"],
        style=estilo_personalizado_selecao
    ).ask()

    if tipo == "📁 Classificar várias imagens (grupo)":
        return classificar_rasters_segmentados()

    modo = questionary.select(
        "Deseja usar ProcessPool (mais pesado) ou ThreadPool (mais leve)?",
        choices=["🔀 ThreadPool (rápido, para modelos leves)", "🔁 ProcessPool (melhor para modelos pesados)"],
        style=estilo_personalizado_selecao
    ).ask()

    modelo_path, raster_entrada, tamanho_bloco, uso_cpu_percentual, n_threads = configurar_classificacao_individual()

    if "ThreadPool" in modo:
        return classificar_imagem_thread(modelo_path, raster_entrada, tamanho_bloco, uso_cpu_percentual, n_threads)
    else:
        return classificar_imagem_pool(modelo_path, raster_entrada, tamanho_bloco, uso_cpu_percentual, n_threads)

## Classificação de imagem em grupo - inicio

def classificar_rasters_segmentados():
    modelo_path = selecionar_arquivo_com_extensoes([".pkl"], mensagem="Selecione o modelo .pkl treinado:")
    raster_exemplo = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione um dos rasters segmentados para definir a pasta:")
    pasta_segmentos = os.path.dirname(raster_exemplo)
    arquivos_tif = [os.path.join(pasta_segmentos, f) for f in sorted(os.listdir(pasta_segmentos)) if f.lower().endswith(".tif")]

    nome_padrao = os.path.basename(pasta_segmentos.rstrip("/\\")) + "_classificados"
    nome_final = questionary.text("Nome da pasta para salvar os rasters classificados:", default=nome_padrao).ask()
    pasta_saida = os.path.join(os.path.dirname(pasta_segmentos), nome_final)
    os.makedirs(pasta_saida, exist_ok=True)

    caminho_log = os.path.join(pasta_saida, "classificados.log")
    caminho_relatorio = os.path.join(pasta_saida, "relatorio_classificacao.txt")

    classificados_existentes = set()
    if os.path.exists(caminho_log):
        with open(caminho_log, "r", encoding="utf-8") as f:
            classificados_existentes = set(linha.strip() for linha in f if linha.strip())

    # Remover arquivos que não estão no log, pois podem estar corrompidos/incompletos
    for f in os.listdir(pasta_saida):
        if f.lower().endswith(".tif") and f not in classificados_existentes:
            try:
                os.remove(os.path.join(pasta_saida, f))
                print(f"[🗑️] Arquivo removido (não consta no log): {f}")
            except Exception as e:
                print(f"[⚠️] Erro ao tentar remover {f}: {e}")

    modelo = load(modelo_path)
    duracoes = []
    arquivos_processados = 0
    total = len(arquivos_tif)

    inicio_geral = None
    total_pixels = 0

    for idx, raster_entrada in enumerate(arquivos_tif):
        nome_base, extensao = os.path.splitext(os.path.basename(raster_entrada))
        nome_saida = f"{nome_base}-classificado{extensao}"
        raster_saida = os.path.join(pasta_saida, nome_saida)

        if nome_saida in classificados_existentes:
            continue

        if inicio_geral is None:
            inicio_geral = time.time()

        inicio = time.time()

        with rasterio.open(raster_entrada) as src:
            profile = src.profile.copy()
            profile.update(dtype='uint8', count=1, compress='lzw')
            largura, altura = src.width, src.height
            nodata = src.nodata

            bloco = src.read().transpose(1, 2, 0)
            mascara_valida = ~(np.all(bloco == 0, axis=2))
            matriz_saida = np.zeros((altura, largura), dtype='uint8')

            if np.any(mascara_valida):
                previsoes = modelo.predict(bloco[mascara_valida])
                matriz_saida[mascara_valida] = previsoes.astype('uint8')

            with rasterio.open(raster_saida, 'w', **profile) as dst:
                dst.write(matriz_saida, 1)

        fim = time.time()
        duracoes.append(fim - inicio)
        arquivos_processados += 1
        total_pixels += largura * altura

        with open(caminho_log, "a", encoding="utf-8") as f:
            f.write(nome_saida + "\n")

        tempo_decorrido = fim - inicio_geral if inicio_geral else 0
        tempo_medio = sum(duracoes) / len(duracoes)
        restantes = total - (len(classificados_existentes) + arquivos_processados)
        estimativa_restante = tempo_medio * restantes
        tempo_estimado_total = tempo_decorrido + estimativa_restante
        percentual = ((len(classificados_existentes) + arquivos_processados) / total) * 100

        Limpar()
        print(f"[🧠] Processamento de rasters segmentado (📁 {os.path.basename(pasta_segmentos)})")
        print(f"    Processado {len(classificados_existentes) + arquivos_processados}/{total} rasters ({percentual:.1f}%)")
        print(f"    Tempo decorrido: {int(tempo_decorrido)}s ({tempo_decorrido/60:.1f}m)")
        print(f"    Tempo estimado até concluir: {int(estimativa_restante)}s ({estimativa_restante/60:.1f}m)")
        print(f"    Tempo total estimado: {int(tempo_estimado_total)}s ({tempo_estimado_total/60:.1f}m)\n")

    if inicio_geral:
        tempo_total = time.time() - inicio_geral
        print("[✔] Classificação concluída.")
        print(f"[📂] Resultados salvos em: {pasta_saida}")
        print(f"[📄] Log salvo em: {caminho_log}")
        print(f"[⏱️] Tempo total decorrido: {tempo_total:.2f} segundos (~{tempo_total/60:.1f} min)")

        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write("RELATÓRIO DE CLASSIFICAÇÃO DE IMAGEM\n")
            f.write(f"Data e hora: {datetime.now()}\n\n")
            f.write(f"Modelo utilizado: {modelo_path}\n\n")
            f.write(f"Tamanho do bloco: imagem completa\n")
            f.write(f"Uso de CPU: 100% (1 processo)\n")
            f.write(f"Total de pixels classificados: {total_pixels}\n")
            f.write(f"Tempo total de execução: {round(tempo_total, 2)} segundos\n")
            f.write(f"Quantidade de rasters processados: {arquivos_processados}\n")
            f.write(f"Pasta de saída: {pasta_saida}\n")

    else:
        print("[✔] Nenhum novo raster precisou ser processado.")

    alerta_conclusao()

## Segmentações e uniões

def segmentar_raster_em_blocos():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="🖼️ Selecione o raster a ser segmentado:")

    escolha = questionary.select(
        "📐 Escolha o tamanho do bloco (em pixels):",
        choices=["256", "512", "1024", "2048", "4096", "8192", "🔧 Inserir valor personalizado"],
        default="2048",
        style=estilo_personalizado_selecao
    ).ask()

    if escolha == "🔧 Inserir valor personalizado":
        while True:
            entrada = questionary.text("🧩 Digite o tamanho do bloco (em pixels):", default="2048").ask()
            if entrada and entrada.isdigit():
                bloco_pixels = int(entrada)
                if bloco_pixels > 0:
                    break
                else:
                    print("[⚠️] Por favor, insira um número positivo maior que zero.")
            else:
                print("[⚠️] Entrada inválida. Insira apenas números inteiros.")
    else:
        bloco_pixels = int(escolha)

    nome_base = os.path.splitext(os.path.basename(raster_path))[0]
    pasta_saida = os.path.join(
        os.path.dirname(raster_path),
        f"{nome_base}-segmentado-{bloco_pixels}"
    )
    os.makedirs(pasta_saida, exist_ok=True)

    print(f"\n🔄 Segmentando o raster com blocos de {bloco_pixels}x{bloco_pixels} pixels...")

    with rasterio.open(raster_path) as src:
        largura, altura = src.width, src.height
        profile = src.profile.copy()

        blocos_total = ((altura + bloco_pixels - 1) // bloco_pixels) * ((largura + bloco_pixels - 1) // bloco_pixels)
        count = 0

        with tqdm(total=blocos_total, desc="📦 Processando blocos", unit="bloco") as barra:
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
                    barra.update(1)

    alerta_conclusao()
    print(f"\n✅ Segmentação concluída.")
    print(f"📦 Total de blocos salvos: {count}")
    print(f"📁 Arquivos armazenados em: {pasta_saida}")

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

    alerta_conclusao()
    print(f"[✔] Mosaico salvo como: {saida_path}")

def segmentar_raster_por_vetor():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster que será segmentado pelas feições do vetor:")
    vetor_path = selecionar_arquivo_com_extensoes([".gpkg"], mensagem="Selecione o vetor com as feições (GPKG):")

    gdf = gpd.read_file(vetor_path)
    campos = [col for col in gdf.columns if col != "geometry"]

    if not campos:
        print("[⚠] Nenhum campo não-geométrico encontrado no vetor.")
        return

    campo_id = questionary.select("Selecione o campo que será usado para nomear os arquivos:", choices=campos).ask()

    base_raster_name = os.path.splitext(os.path.basename(raster_path))[0]
    saida_dir = os.path.join(os.path.dirname(raster_path), f"{base_raster_name}_segmentado_por_vetor")
    os.makedirs(saida_dir, exist_ok=True)

    with rasterio.open(raster_path) as src:
        for idx, row in gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, out_transform = mask(src, geom, crop=True)
            except Exception as e:
                print(f"[❌] Erro ao recortar feição {idx}: {e}")
                continue

            if (out_image == 0).all():
                print(f"[⚠] Segmento ignorado (somente valores 0) para feição {idx}.")
                continue

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            valor_base = str(row[campo_id]).replace("/", "-").replace(" ", "_")
            nome_arquivo = f"{campo_id}_{valor_base}.tif"
            saida_path = os.path.join(saida_dir, nome_arquivo)
            contador = 2
            while os.path.exists(saida_path):
                nome_arquivo = f"{campo_id}_{valor_base}-{contador}.tif"
                saida_path = os.path.join(saida_dir, nome_arquivo)
                contador += 1

            with rasterio.open(saida_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"[✔] Segmento salvo: {saida_path}")

    alerta_conclusao()
    print(f"[✔] Segmentos salvos na pasta: {saida_dir}")

## Analises
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

        alerta_conclusao()
        print(f"[📊] Relatório gerado: {relatorio_path}")

def gerar_relatorio_area_segmentos():
    pasta_segmentos = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione um dos rasters segmentados para definir a pasta:")
    pasta_segmentos = os.path.dirname(pasta_segmentos)
    arquivos_tif = [f for f in os.listdir(pasta_segmentos) if f.lower().endswith(".tif")]

    if not arquivos_tif:
        print("[⚠] Nenhum arquivo .tif encontrado na pasta.")
        return

    print(f"[📂] Analisando {len(arquivos_tif)} arquivos .tif na pasta: {pasta_segmentos}")
    relatorio = []
    todas_classes = set()
    registros_por_arquivo = {}

    for arquivo in tqdm(arquivos_tif, desc="Processando rasters"):
        caminho = os.path.join(pasta_segmentos, arquivo)
        with rasterio.open(caminho) as src:
            array = src.read(1)
            transform = src.transform
            res_x, res_y = transform[0], -transform[4]
            area_pixel_km2 = (res_x * res_y) / 1_000_000

            classes, contagens = np.unique(array, return_counts=True)
            total_pixels = 0
            info = {"arquivo": arquivo}

            for classe, contagem in zip(classes, contagens):
                if classe == 0:
                    continue
                todas_classes.add(classe)
                info[f"{classe}_px"] = contagem
                info[f"{classe}_km2"] = round(contagem * area_pixel_km2, 4)
                total_pixels += contagem

            info["total_px"] = total_pixels
            info["total_km2"] = round(total_pixels * area_pixel_km2, 4)

            registros_por_arquivo[arquivo] = info

    # Cálculo de percentuais após conhecer todas as classes
    for info in registros_por_arquivo.values():
        total = info.get("total_px", 0)
        for classe in todas_classes:
            key_px = f"{classe}_px"
            key_pct = f"{classe}_pct"
            if key_px in info:
                info[key_pct] = round(100 * info[key_px] / total, 2)
            else:
                info[f"{classe}_px"] = 0
                info[f"{classe}_km2"] = 0.0
                info[key_pct] = 0.0
        relatorio.append(info)

    colunas = ["arquivo"]
    for classe in sorted(todas_classes):
        colunas.append(f"{classe}_px")
    colunas.append("total_px")
    for classe in sorted(todas_classes):
        colunas.append(f"{classe}_km2")
    colunas.append("total_km2")
    for classe in sorted(todas_classes):
        colunas.append(f"{classe}_pct")

    df = pd.DataFrame(relatorio)[colunas]
    nome_saida = os.path.join(pasta_segmentos, "relatorio_areas_segmentadas.xlsx")
    df.to_excel(nome_saida, index=False)
    print(f"[📄] Relatório salvo em: {nome_saida}")

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
            default="Sim", 
            style=estilo_personalizado_selecao
        ).ask() == "Sim"

        salvar_raster_diferencas = questionary.select(
            "Deseja gerar o raster de diferenças?",
            choices=["Sim", "Não"],
            default="Sim", 
            style=estilo_personalizado_selecao
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

        alerta_conclusao()
        print("\n[✔] Comparação concluída.")

def verificar_resolucao_raster():
    modo = questionary.select(
        "Deseja verificar a resolução de um único raster ou de todos os rasters em uma pasta?",
        choices=["📄 Analisar um único raster", "📁 Analisar todos os rasters de uma pasta"],
        style=estilo_personalizado_selecao
    ).ask()

    resultados = []

    if modo == "📄 Analisar um único raster":
        raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster para verificar a resolução:")
        with rasterio.open(raster_path) as src:
            transform = src.transform
            res_x, res_y = transform[0], -transform[4]
        resultados.append((os.path.basename(raster_path), res_x, res_y))

    else:
        raster_exemplo = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione qualquer raster da pasta desejada:")
        pasta = os.path.dirname(raster_exemplo)
        arquivos_tif = [f for f in sorted(os.listdir(pasta)) if f.lower().endswith(".tif")]

        for nome in arquivos_tif:
            caminho = os.path.join(pasta, nome)
            try:
                with rasterio.open(caminho) as src:
                    transform = src.transform
                    res_x, res_y = transform[0], -transform[4]
                resultados.append((nome, res_x, res_y))
            except:
                resultados.append((nome, None, None))

    # Exibir resumo
    print("\n📊 RESUMO DAS RESOLUÇÕES:")
    for nome, res_x, res_y in resultados:
        if res_x is not None:
            print(f"🗂️ {nome} → {res_x:.4f}m x {res_y:.4f}m (área: {res_x * res_y:.4f} m²)")
        else:
            print(f"⚠️ {nome} → Erro ao abrir ou ler o raster.")

    alerta_conclusao()

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

    alerta_conclusao()
    print(f"[✔] Imagem salva sem banda alfa como: {saida}")

## Redução de ruído

def modo_local(pixels, nodata):
    try:
        valores = pixels[pixels != nodata] if nodata is not None else pixels
        valores = valores[valores >= 0]  # evita valores negativos para bincount
        if len(valores) == 0:
            return nodata if nodata is not None else 0
        return np.bincount(valores.astype(int)).argmax()
    except Exception as e:
        print(f"[⚠️ ERRO modo_local] {e}")
        return nodata if nodata is not None else 0

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

def aplicar_filtro_modo_individual():
    raster_path = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster classificado para aplicar o filtro:")
    if not raster_path:
        return

    tamanho_janela = int(questionary.text("Tamanho da janela (ímpar, ex: 3, 5, 7):", default="3").ask())
    if tamanho_janela % 2 == 0:
        print("[⚠] A janela deve ser um número ímpar.")
        return

    bloco_altura = int(questionary.select(
        "Altura do bloco (linhas por processo):",
        choices=["128", "256", "512", "1024", "2048", "4096"],
        default="1024",
        style=estilo_personalizado_selecao
    ).ask())

    total_cores = mp.cpu_count()
    uso_cpu = int(questionary.text(f"Uso da CPU (%):", default="100").ask())
    n_proc = max(1, int((uso_cpu / 100) * total_cores))

    with rasterio.open(raster_path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
        altura, largura = array.shape

    blocos = [
        (array, i, min(i + bloco_altura, altura), nodata, tamanho_janela)
        for i in range(0, altura, bloco_altura)
    ]
    array_filtrado = np.full_like(array, nodata if nodata is not None else 0)

    inicio = time.time()
    with mp.Pool(processes=n_proc) as pool:
        for inicio_bloco, resultado in pool.imap(processar_bloco_vertical, blocos):
            array_filtrado[inicio_bloco:inicio_bloco + resultado.shape[0], :] = resultado

    profile.update(compress='lzw')
    if nodata is not None:
        profile.update(nodata=nodata)

    nome_base = os.path.splitext(os.path.basename(raster_path))[0]
    saida_path = os.path.join(os.path.dirname(raster_path), f"{nome_base}_modo{tamanho_janela}.tif")

    contador = 2
    while os.path.exists(saida_path):
        saida_path = os.path.join(os.path.dirname(raster_path), f"{nome_base}_modo{tamanho_janela}({contador}).tif")
        contador += 1

    with rasterio.open(saida_path, "w", **profile) as dst:
        dst.write(array_filtrado, 1)

    alerta_conclusao()
    tempo_total = time.time() - inicio
    print(f"\n[✅] Filtro aplicado. Arquivo salvo em: {saida_path}")
    print(f"[⏱] Tempo total: {tempo_total:.2f} segundos")

def wrapper_filtro_raster(args):
    raster_path, pasta_saida, tamanho_janela = args
    try:
        with rasterio.open(raster_path) as src:
            array = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata
            altura, largura = array.shape

        array_filtrado = generic_filter(
            array,
            lambda p: modo_local(p, nodata),
            size=tamanho_janela,
            mode="nearest"
        )

        nome_base = os.path.splitext(os.path.basename(raster_path))[0]
        saida_path = os.path.join(pasta_saida, f"{nome_base}_modo{tamanho_janela}.tif")
        profile.update(compress='lzw')
        if nodata is not None:
            profile.update(nodata=nodata)

        with rasterio.open(saida_path, "w", **profile) as dst:
            dst.write(array_filtrado, 1)

        return f"[✔] {os.path.basename(raster_path)} filtrado"
    except Exception as e:
        return f"[⚠️] Erro em {raster_path}: {e}"

def aplicar_filtro_modo_lote_por_arquivo():
    raster_exemplo = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione um dos rasters classificados:")
    pasta_origem = os.path.dirname(raster_exemplo)
    arquivos = [os.path.join(pasta_origem, f) for f in os.listdir(pasta_origem) if f.lower().endswith(".tif")]

    tamanho_janela = int(questionary.text("Tamanho da janela (ímpar, ex: 3, 5, 7):", default="3").ask())
    if tamanho_janela % 2 == 0:
        print("[⚠] A janela deve ser um número ímpar.")
        return

    total_cores = mp.cpu_count()
    uso_cpu = int(questionary.text(f"Uso da CPU (%):", default="100").ask())
    n_proc = max(1, int((uso_cpu / 100) * total_cores))

    nome_pasta = os.path.basename(pasta_origem.rstrip("/\\")) + f"-modo{tamanho_janela}"
    pasta_saida = os.path.join(os.path.dirname(pasta_origem), nome_pasta)
    os.makedirs(pasta_saida, exist_ok=True)

    # ✅ Carrega log de arquivos já processados
    caminho_log = os.path.join(pasta_saida, "processados.log")
    processados = set()
    if os.path.exists(caminho_log):
        with open(caminho_log, "r", encoding="utf-8") as f:
            processados = set(linha.strip() for linha in f if linha.strip())

    # ✅ Monta tarefas somente com rasters ainda não processados
    tarefas = []
    nomes_saida = []
    for raster_path in arquivos:
        nome_base = os.path.splitext(os.path.basename(raster_path))[0]
        nome_saida = f"{nome_base}_modo{tamanho_janela}.tif"
        if nome_saida not in processados:
            tarefas.append((raster_path, pasta_saida, tamanho_janela))
            nomes_saida.append(nome_saida)

    total = len(tarefas)
    if total == 0:
        print("[ℹ] Todos os rasters já foram processados anteriormente.")
        return

    print(f"\n🔁 Iniciando limpeza de ruído em {total} raster(s) com {n_proc} processo(s)...\n")

    inicio_total = time.time()
    duracoes = []
    concluídos = 0

    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        for idx, resultado in enumerate(tqdm(executor.map(wrapper_filtro_raster, tarefas), total=total, desc="🧼 Filtrando")):
            nome_saida = nomes_saida[idx]
            if "[✔]" in resultado:
                with open(caminho_log, "a", encoding="utf-8") as f:
                    f.write(nome_saida + "\n")

            print(resultado)

            concluídos += 1
            tempo_decorrido = int(time.time() - inicio_total)
            tempo_medio = tempo_decorrido / concluídos
            restante = total - concluídos
            estimado_restante = int(tempo_medio * restante)
            estimado_total = int(tempo_decorrido + estimado_restante)

            Limpar()
            print(f"\n📊 LIMPEZA DE RUÍDO EM LOTE")
            print(f"Rasters processados: {concluídos}/{total}")
            print(f"Tempo decorrido: {tempo_decorrido}s ({tempo_decorrido // 60} min)")
            print(f"Tempo estimado para finalizar: {estimado_restante}s ({estimado_restante // 60} min)")
            print(f"Tempo total estimado: {estimado_total}s ({estimado_total // 60} min)\n")

    alerta_conclusao()
    tempo_total = int(time.time() - inicio_total)
    print(f"[✅] Todos os filtros foram aplicados. Rasters salvos em: {pasta_saida}")
    print(f"[📄] Log salvo em: {caminho_log}")
    print(f"[⏱] Tempo total de execução: {tempo_total}s ({tempo_total // 60} min)")

def aplicar_filtro_modo_lote_por_blocos():
    raster_exemplo = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione um dos rasters classificados:")
    pasta_origem = os.path.dirname(raster_exemplo)
    arquivos = [os.path.join(pasta_origem, f) for f in os.listdir(pasta_origem) if f.lower().endswith(".tif")]

    tamanho_janela = int(questionary.text("Tamanho da janela (ímpar, ex: 3, 5, 7):", default="3").ask())
    if tamanho_janela % 2 == 0:
        print("[⚠] A janela deve ser um número ímpar.")
        return

    bloco_altura = int(questionary.select(
        "Altura do bloco (linhas por processo):",
        choices=["128", "256", "512", "1024", "2048", "4096"],
        default="1024",
        style=estilo_personalizado_selecao
    ).ask())

    total_cores = mp.cpu_count()
    uso_cpu = int(questionary.text(f"Uso da CPU (%):", default="100").ask())
    n_proc = max(1, int((uso_cpu / 100) * total_cores))

    nome_pasta = os.path.basename(pasta_origem.rstrip("/\\")) + f"-modo{tamanho_janela}"
    pasta_saida = os.path.join(os.path.dirname(pasta_origem), nome_pasta)
    os.makedirs(pasta_saida, exist_ok=True)

    # Log
    caminho_log = os.path.join(pasta_saida, "processados.log")
    processados = set()
    if os.path.exists(caminho_log):
        with open(caminho_log, "r", encoding="utf-8") as f:
            processados = set(linha.strip() for linha in f if linha.strip())

    tarefas = []
    for raster_path in arquivos:
        nome_base = os.path.splitext(os.path.basename(raster_path))[0]
        nome_saida = f"{nome_base}_modo{tamanho_janela}.tif"
        if nome_saida not in processados:
            tarefas.append((raster_path, nome_saida))

    total = len(tarefas)
    if total == 0:
        print("[ℹ] Todos os rasters já foram processados anteriormente.")
        return

    print(f"\n🔁 Iniciando limpeza de ruído por blocos em {total} raster(s) com {n_proc} processo(s)...\n")

    inicio_total = time.time()
    duracoes = []
    concluídos = 0

    for idx, (raster_path, nome_saida) in enumerate(tarefas):
        t_inicio = time.time()

        try:
            with rasterio.open(raster_path) as src:
                array = src.read(1)
                profile = src.profile.copy()
                nodata = src.nodata
                altura, largura = array.shape

            blocos = [
                (array, i, min(i + bloco_altura, altura), nodata, tamanho_janela)
                for i in range(0, altura, bloco_altura)
            ]
            array_filtrado = np.full_like(array, nodata if nodata is not None else 0)

            with mp.Pool(processes=n_proc) as pool:
                for inicio_bloco, resultado in pool.imap(processar_bloco_vertical, blocos):
                    array_filtrado[inicio_bloco:inicio_bloco + resultado.shape[0], :] = resultado

            profile.update(compress='lzw')
            if nodata is not None:
                profile.update(nodata=nodata)

            saida_path = os.path.join(pasta_saida, nome_saida)
            contador = 2
            while os.path.exists(saida_path):
                saida_path = os.path.join(pasta_saida, f"{os.path.splitext(nome_saida)[0]}({contador}).tif")
                contador += 1

            with rasterio.open(saida_path, "w", **profile) as dst:
                dst.write(array_filtrado, 1)

            with open(caminho_log, "a", encoding="utf-8") as f:
                f.write(os.path.basename(saida_path) + "\n")

            print(f"[✔] {os.path.basename(raster_path)} processado com sucesso.")

        except Exception as e:
            print(f"[⚠️] Erro ao processar {raster_path}: {e}")
            continue

        # Resumo após cada raster
        concluídos += 1
        tempo_raster = time.time() - t_inicio
        tempo_decorrido = time.time() - inicio_total
        duracoes.append(tempo_raster)
        tempo_medio = sum(duracoes) / len(duracoes)
        restante = total - concluídos
        estimado_restante = int(tempo_medio * restante)
        estimado_total = int(tempo_decorrido + estimado_restante)

        Limpar()
        print(f"\n📊 LIMPEZA DE RUÍDO POR BLOCOS")
        print(f"Rasters processados: {concluídos}/{total}")
        print(f"Tempo decorrido: {int(tempo_decorrido)}s ({tempo_decorrido // 60} min)")
        print(f"Tempo estimado para finalizar: {estimado_restante}s ({estimado_restante // 60} min)")
        print(f"Tempo total estimado: {estimado_total}s ({estimado_total // 60} min)\n")

    alerta_conclusao()
    tempo_total = int(time.time() - inicio_total)
    print(f"[✅] Todos os filtros foram aplicados. Rasters salvos em: {pasta_saida}")
    print(f"[📄] Log salvo em: {caminho_log}")
    print(f"[⏱] Tempo total de execução: {tempo_total}s ({tempo_total // 60} min)")

## Matrizes de confusão

def gerar_matriz_confusao_raster():
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Importado na função para deixar o inicio mais leve

    print("[📊] Gerar matriz de confusão entre raster de referência e classificado")
    raster_ref = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster de referência (verdade do terreno):")
    raster_pred = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster classificado (modelo predito):")

    with rasterio.open(raster_ref) as ref, rasterio.open(raster_pred) as pred:
        if ref.shape != pred.shape or ref.transform != pred.transform:
            print("[❌] Os rasters não têm a mesma dimensão ou alinhamento espacial.")
            return

        ref_array = ref.read(1)
        pred_array = pred.read(1)
        nodata_ref = ref.nodata
        nodata_pred = pred.nodata

        mask_valid = np.ones(ref_array.shape, dtype=bool)
        if nodata_ref is not None:
            mask_valid &= ref_array != nodata_ref
        if nodata_pred is not None:
            mask_valid &= pred_array != nodata_pred

        y_true = ref_array[mask_valid].flatten()
        y_pred = pred_array[mask_valid].flatten()

        if len(y_true) == 0:
            print("[⚠] Nenhum pixel válido para comparação.")
            return

        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        matriz = confusion_matrix(y_true, y_pred, labels=labels)
        acuracia = accuracy_score(y_true, y_pred)
        relatorio = classification_report(y_true, y_pred, labels=labels, output_dict=True)

        print("\n[✅] Matriz de Confusão:")
        print(matriz)
        print(f"\n[🎯] Acurácia Global: {acuracia:.4f} ({acuracia*100:.2f}%)")
        print("\n[🧾] Acurácia por Classe:")
        for label in labels:
            acc_label = relatorio[str(label)]['recall']
            print(f"Classe {label}: {acc_label:.4f} ({acc_label*100:.2f}%)")

        print("\n[🧾] Relatório de Classificação:")
        print(classification_report(y_true, y_pred, labels=labels))

        data_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_saida = f"matriz_confusao_{data_str}.txt"
        with open(nome_saida, "w", encoding="utf-8") as f:
            f.write("MATRIZ DE CONFUSÃO\n")
            f.write(f"Referência: {raster_ref}\n")
            f.write(f"Classificado: {raster_pred}\n")
            f.write(f"Data: {datetime.now()}\n\n")
            f.write("Classes analisadas: " + ", ".join(map(str, labels)) + "\n\n")
            f.write(str(matriz))
            f.write(f"\n\nAcurácia Global: {acuracia:.4f} ({acuracia*100:.2f}%)\n")
            f.write("\nAcurácia por Classe:\n")
            for label in labels:
                acc_label = relatorio[str(label)]['recall']
                f.write(f"Classe {label}: {acc_label:.4f} ({acc_label*100:.2f}%)\n")
            f.write("\nRELATÓRIO:\n")
            f.write(classification_report(y_true, y_pred, labels=labels))

        alerta_conclusao()
        print(f"\n[💾] Relatório salvo como: {nome_saida}")

def gerar_matriz_confusao_vetor():
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Importado na função para deixar o inicio mais leve

    print("[📊] Gerar matriz de confusão com base em vetor de amostras")
    raster_pred = selecionar_arquivo_com_extensoes([".tif"], mensagem="Selecione o raster classificado (modelo predito):")
    vetor_ref = selecionar_arquivo_com_extensoes([".gpkg"], mensagem="Selecione o arquivo de amostras (GPKG):")
    campo_classe = questionary.text("Nome do campo de classe no vetor:").ask()

    with rasterio.open(raster_pred) as src:
        perfil = src.profile
        transform = src.transform
        shape = src.shape
        pred_array = src.read(1)
        nodata_pred = src.nodata

    gdf = gpd.read_file(vetor_ref)
    gdf = gdf[[campo_classe, "geometry"]].dropna(subset=["geometry"])

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[campo_classe]))
    ref_array = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=-9999,
        dtype="int32"
    )

    mask_valid = (ref_array != -9999)
    if nodata_pred is not None:
        mask_valid &= pred_array != nodata_pred

    y_true = ref_array[mask_valid].flatten()
    y_pred = pred_array[mask_valid].flatten()

    if len(y_true) == 0:
        print("[⚠] Nenhum pixel válido para comparação.")
        return

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    acuracia = accuracy_score(y_true, y_pred)
    relatorio = classification_report(y_true, y_pred, labels=labels, output_dict=True)

    print("\n[✅] Matriz de Confusão:")
    print(matriz)
    print(f"\n[🎯] Acurácia Global: {acuracia:.4f} ({acuracia*100:.2f}%)")
    print("\n[🧾] Acurácia por Classe:")
    for label in labels:
        acc_label = relatorio[str(label)]['recall']
        print(f"Classe {label}: {acc_label:.4f} ({acc_label*100:.2f}%)")

    print("\n[🧾] Relatório de Classificação:")
    print(classification_report(y_true, y_pred, labels=labels))

    data_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_saida = f"matriz_confusao_vetor_{data_str}.txt"
    with open(nome_saida, "w", encoding="utf-8") as f:
        f.write("MATRIZ DE CONFUSÃO - REFERÊNCIA VETORIAL\n")
        f.write(f"Raster classificado: {raster_pred}\n")
        f.write(f"Vetor referência: {vetor_ref}\n")
        f.write(f"Campo de classe: {campo_classe}\n")
        f.write(f"Data: {datetime.now()}\n\n")
        f.write("Classes analisadas: " + ", ".join(map(str, labels)) + "\n\n")
        f.write(str(matriz))
        f.write(f"\n\nAcurácia Global: {acuracia:.4f} ({acuracia*100:.2f}%)\n")
        f.write("\nAcurácia por Classe:\n")
        for label in labels:
            acc_label = relatorio[str(label)]['recall']
            f.write(f"Classe {label}: {acc_label:.4f} ({acc_label*100:.2f}%)\n")
        f.write("\nRELATÓRIO:\n")
        f.write(classification_report(y_true, y_pred, labels=labels))

    alerta_conclusao()
    print(f"\n[💾] Relatório salvo como: {nome_saida}")

# === Submenus ===
def submenu(titulo, opcoes):
    opcoes_submenu = opcoes + [("🔙 Voltar", None)]
    while True:
        escolha = questionary.select(
            titulo,
            choices=[texto for texto, _ in opcoes_submenu],
            style=estilo_personalizado_selecao
        ).ask()

        for texto, funcao in opcoes_submenu:
            if escolha == texto:
                Limpar()
                if funcao is None:
                    return
                funcao()
                break

def submenu_segmentacao():
    titulo = "🧩 Segmentação e unificação - Escolha uma ação:"
    opcoes_submenu = [
        ("🧩 Segmentar rasters em blocos fixos", segmentar_raster_em_blocos),
        ("🧩 Segmentar rasters com vetores", segmentar_raster_por_vetor),
        ("🖼️ Unificar rasters", unir_rasters_em_mosaico),
    ]
    submenu(titulo, opcoes_submenu)

def submenu_matriz():
    titulo = "📊 Matrizes de confusão - Escolha uma ação:"
    opcoes_submenu = [
        ("📊 Matriz de confusão (Raster x Raster)", gerar_matriz_confusao_raster),
        ("📊 Matriz de confusão (Raster x Vetor)", gerar_matriz_confusao_vetor),
    ]
    submenu(titulo, opcoes_submenu)

def submenu_analisar():
    titulo = "🔍 Análises e verificações - Escolha uma ação:"
    opcoes_submenu = [
        ("🔎 Analisar raster", analisar_raster),
        ("🔎 Analisar raster em grupo", gerar_relatorio_area_segmentos),
        ("📐 Verificar resolução do raster", verificar_resolucao_raster),
        ("🖼️ Comparar rasters", comparar_rasters),
    ]
    submenu(titulo, opcoes_submenu)

def submenu_limpar_ruido():
    titulo = "🧼 Redução de Ruído - Escolha uma opção:"
    opcoes_submenu = [
        ("📄 Aplicar filtro em um único raster", aplicar_filtro_modo_individual),
        ("📁 Aplicar filtro em lote (Paralelo por bloco interno)", aplicar_filtro_modo_lote_por_blocos),
        ("📁 Aplicar filtro em lote (Paralelo por Arquivo)", aplicar_filtro_modo_lote_por_arquivo),
    ]
    submenu(titulo, opcoes_submenu)

# === Menu principal ===
def menu():
    exibir_banner()
    opcoes = [
        ("🧠 Treinar modelo", treinar_modelo),
        ("🧮 Classificar raster (individual ou em grupo)", classificar_imagem),
        ("🧼 Limpar ruído", submenu_limpar_ruido),
        ("🧩 Segmentação e unificação de raster", submenu_segmentacao),
        ("🔎 Analisar raster", submenu_analisar),
        ("📊 Matrizes de confusão", submenu_matriz),
        ("🧹 Remover banda 4 (imagem RGB)", remover_banda_4),
        ("🧹 Limpar prompt", Limpar),
        ("❌ Sair", None)
    ]
    
    while True:
        escolha = questionary.select(
            "📋 MENU PRINCIPAL - Escolha uma ação:",
            choices=[texto for texto, _ in opcoes],
            style=estilo_personalizado_selecao
        ).ask()

        for texto, funcao in opcoes:
            if escolha == texto:
                if funcao is None:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    return
                Limpar()
                funcao()
                break

if __name__ == "__main__":
    Limpar(banner=False)
    menu()
