#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROJETO BIG DATA & HADOOP — CORRELAÇÃO ESTATÍSTICA DE PEARSON VIA MAPREDUCE
Dataset: Forest Fires (Incêndios Florestais - UCI Machine Learning Repository)
Objetivo: Calcular a correlação de Pearson entre a Coluna 9 (Temperatura - temp)
          e todas as demais colunas numéricas do dataset, garantindo integridade
          atômica dos dados nos nós escravos (Mappers / Combiners / Reducers).
================================================================================
"""

import sys
import csv
import math
import json
import argparse
from collections import defaultdict
from typing import Dict, Generator, Iterable, List, Optional, Tuple


# ==============================================================================
# 1. DEFINIÇÃO DO ESQUEMA E METADADOS DO DATASET (FOREST FIRES)
# ==============================================================================

TOTAL_COLUNAS_ESPERADAS = 13
INDICE_COLUNA_ALVO = 8  # Coluna 9 (1-based) -> Índice 8 (0-based): 'temp' (Temperatura em °C)
NOME_COLUNA_ALVO = "Coluna 09 - temp (Temperatura °C)"

# Mapeamento das demais colunas numéricas do dataset (0-based index -> Nome descritivo)
COLUNAS_NUMERICAS_COMPARACAO: Dict[int, str] = {
    0: "Coluna 01 - X (Coordenada Espacial X)",
    1: "Coluna 02 - Y (Coordenada Espacial Y)",
    4: "Coluna 05 - FFMC (Índice de Umidade de Combustível Fino)",
    5: "Coluna 06 - DMC (Índice de Umidade da Camada Orgânica)",
    6: "Coluna 07 - DC (Índice de Seca / Drought Code)",
    7: "Coluna 08 - ISI (Índice de Propagação Inicial)",
    9: "Coluna 10 - RH (Umidade Relativa %)",
    10: "Coluna 11 - wind (Velocidade do Vento km/h)",
    11: "Coluna 12 - rain (Precipitação mm/m²)",
    12: "Coluna 13 - area (Área Queimada em hectares)",
}

# Conjuntos de domínio válido para validação de integridade das colunas categóricas
MESES_VALIDOS = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
}
DIAS_VALIDOS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}

# Marcadores textuais considerados nulos ou ausentes em bases sujas
MARCADORES_NULOS = {"", "null", "none", "na", "n/a", "nan", "?", "-", "undefined"}

# Limites físicos plausíveis (Domínio de Integridade) para evitar outliers corrompidos
LIMITES_DOMINIO_FISICO: Dict[int, Tuple[float, float]] = {
    0: (1.0, 9.0),         # X: mapa do Parque Montesinho (1 a 9)
    1: (1.0, 9.0),         # Y: mapa do Parque Montesinho (1 a 9)
    4: (0.0, 101.0),       # FFMC: escala FWI (0 a 101)
    5: (0.0, 1000.0),      # DMC: escala FWI (>= 0)
    6: (0.0, 2000.0),      # DC: escala FWI (>= 0)
    7: (0.0, 200.0),       # ISI: escala FWI (>= 0)
    8: (-50.0, 65.0),      # temp: Temperatura terrestre plausível em °C
    9: (0.0, 100.0),       # RH: Umidade relativa em % (0 a 100)
    10: (0.0, 200.0),      # wind: Velocidade do vento em km/h (>= 0)
    11: (0.0, 500.0),      # rain: Chuva em mm/m² (>= 0)
    12: (0.0, 100000.0),   # area: Área queimada em ha (>= 0)
}

# Tipo para a tupla de estatísticas suficientes: (n, sum_x, sum_y, sum_x2, sum_y2, sum_xy)
VetorSuficiente = Tuple[int, float, float, float, float, float]


# ==============================================================================
# 2. CAMADA DE VALIDAÇÃO E INTEGRIDADE DE DADOS (PRÉ-PROCESSAMENTO NO MAPPER)
# ==============================================================================

def validar_e_extrair_linha(linha_bruta: str) -> Tuple[bool, Optional[Dict[int, float]], str]:
    """
    Realiza a validação atômica e rigorosa de uma linha do dataset antes que
    qualquer dado seja emitido pelos nós escravos (Mappers) para o Shuffle/Reducer.

    Regras de Integridade Aplicadas:
    1. Sanitização de caracteres invisíveis (BOM, CR/LF, espaços laterais).
    2. Descarte seguro de linhas vazias e da linha de cabeçalho original.
    3. Verificação estrutural de schema (exatamente 13 colunas separadas por vírgula).
    4. Detecção de campos nulos, vazios ou marcadores de ausência ('NA', 'null', '?').
    5. Validação semântica das colunas categóricas (month e day).
    6. Conversão numérica estrita (float), verificação de finitude (anti-NaN/Inf)
       e validação de limites físicos de domínio em TODAS as colunas numéricas.

    Retorna:
        (True, dicionario_valores_numericos, "OK") se a linha for 100% íntegra.
        (False, None, motivo_rejeicao) se houver qualquer anomalia.
    """
    if not linha_bruta:
        return False, None, "LINHA_VAZIA"

    # Remove BOM UTF-8 (se existir) e espaços/quebras de linha nas extremidades
    linha_limpa = linha_bruta.lstrip("\ufeff").strip()
    if not linha_limpa:
        return False, None, "LINHA_EM_BRANCO"

    # Parsing seguro de CSV tratando eventuais aspas ou formatações complexas
    try:
        leitor = csv.reader([linha_limpa], delimiter=",")
        colunas = [campo.strip() for campo in next(leitor)]
    except Exception:
        return False, None, "ERRO_PARSING_CSV"

    # Identifica e ignora o cabeçalho do arquivo CSV
    if colunas[0].lower() == "x" and len(colunas) >= 9 and colunas[8].lower() == "temp":
        return False, None, "CABECALHO_CSV"

    # 1. Validação Estrutural: garante que a linha não foi truncada nem possui colunas extras
    if len(colunas) != TOTAL_COLUNAS_ESPERADAS:
        return False, None, f"QUANTIDADE_COLUNAS_INVALIDA({len(colunas)}!={TOTAL_COLUNAS_ESPERADAS})"

    # 2. Validação de Valores Nulos ou Marcadores Inválidos em todas as 13 colunas
    for indice, valor_texto in enumerate(colunas):
        if valor_texto.lower() in MARCADORES_NULOS:
            return False, None, f"VALOR_NULO_COLUNA_{indice + 1}"

    # 3. Validação das Colunas Categóricas (Coluna 3: month [idx 2], Coluna 4: day [idx 3])
    mes = colunas[2].lower()
    dia = colunas[3].lower()
    if mes not in MESES_VALIDOS:
        return False, None, f"MES_INVALIDO({colunas[2]})"
    if dia not in DIAS_VALIDOS:
        return False, None, f"DIA_INVALIDO({colunas[3]})"

    # 4. Validação Numérica, Finitude e Domínio Físico (Coluna 9 + demais colunas numéricas)
    valores_numericos: Dict[int, float] = {}
    indices_numericos = [INDICE_COLUNA_ALVO] + list(COLUNAS_NUMERICAS_COMPARACAO.keys())

    for idx in indices_numericos:
        texto_num = colunas[idx].replace(",", ".")  # Previne erro caso decimal venha com vírgula
        try:
            valor_float = float(texto_num)
        except ValueError:
            return False, None, f"VALOR_MALFORMATADO_COLUNA_{idx + 1}({colunas[idx]})"

        # Impede que valores NaN ou Infinito contaminem os acumuladores do Reducer
        if not math.isfinite(valor_float):
            return False, None, f"VALOR_NAO_FINITO_COLUNA_{idx + 1}"

        # Verifica se o valor está dentro dos limites físicos plausíveis
        limite_min, limite_max = LIMITES_DOMINIO_FISICO[idx]
        if not (limite_min <= valor_float <= limite_max):
            return False, None, f"VALOR_FORA_DO_DOMINIO_COLUNA_{idx + 1}({valor_float})"

        valores_numericos[idx] = valor_float

    return True, valores_numericos, "OK"


# ==============================================================================
# 3. FUNÇÕES DO PARADIGMA MAPREDUCE (MAPPER, COMBINER E REDUCER)
# ==============================================================================

def mapper_linha(linha: str) -> Generator[Tuple[str, VetorSuficiente], None, None]:
    """
    ETAPA MAP (Executada nos nós escravos / Mappers):
    Recebe uma linha bruta do HDFS/entrada, valida sua integridade total e,
    se aprovada, extrai X = Coluna 9 (temp) e Y = cada uma das demais colunas
    numéricas, emitindo as estatísticas suficientes de 1ª e 2ª ordem:
        Chave: Nome identificador da coluna Y
        Valor: (1, x, y, x², y², x*y)
    """
    integra, valores, motivo = validar_e_extrair_linha(linha)

    if not integra or valores is None:
        # Emite métrica de auditoria para monitorar linhas descartadas sem interromper o cluster
        if motivo != "CABECALHO_CSV" and motivo != "LINHA_VAZIA":
            yield ("__AUDITORIA_DESCARTADAS__", (1, 0.0, 0.0, 0.0, 0.0, 0.0))
        return

    # Variável Base X: Coluna 9 (Temperatura em °C - índice 8)
    x = valores[INDICE_COLUNA_ALVO]
    x2 = x * x

    # Para cada uma das demais 10 colunas numéricas (Variável Y), emite o par Chave-Valor
    for idx_y, nome_coluna_y in COLUNAS_NUMERICAS_COMPARACAO.items():
        y = valores[idx_y]
        y2 = y * y
        xy = x * y

        # Emissão atômica: (n=1, sum_x=x, sum_y=y, sum_x2=x², sum_y2=y², sum_xy=x*y)
        yield (nome_coluna_y, (1, x, y, x2, y2, xy))


def combiner_parcial(
    chave: str, valores: Iterable[VetorSuficiente]
) -> Tuple[str, VetorSuficiente]:
    """
    ETAPA COMBINER (Mini-Reducer local no nó Mapper):
    Soma localmente as tuplas (n, x, y, x², y², xy) de uma mesma chave antes
    do tráfego de rede (Shuffle). Como a adição é associativa e comutativa,
    reduz drasticamente o volume de pacotes trafegados entre os DataNodes.
    """
    soma_n = 0
    soma_x = 0.0
    soma_y = 0.0
    soma_x2 = 0.0
    soma_y2 = 0.0
    soma_xy = 0.0

    for n, x, y, x2, y2, xy in valores:
        soma_n += int(n)
        soma_x += float(x)
        soma_y += float(y)
        soma_x2 += float(x2)
        soma_y2 += float(y2)
        soma_xy += float(xy)

    return chave, (soma_n, soma_x, soma_y, soma_x2, soma_y2, soma_xy)


def calcular_pearson_seguro(
    n: int, soma_x: float, soma_y: float, soma_x2: float, soma_y2: float, soma_xy: float
) -> Tuple[float, str]:
    """
    Calcula o Coeficiente de Correlação Linear de Pearson (r) a partir dos
    somatórios acumulados no Reducer:

                 N * Σ(xy) - (Σx)*(Σy)
        r = -------------------------------------------------------
            sqrt( [N * Σ(x²) - (Σx)²] * [N * Σ(y²) - (Σy)²] )

    Inclui proteção contra divisão por zero (variância nula) e truncamento de
    arredondamento IEEE 754 no intervalo [-1.0, +1.0].
    """
    if n < 2:
        return 0.0, "Amostra insuficiente (N < 2)"

    numerador = (n * soma_xy) - (soma_x * soma_y)
    termo_variancia_x = (n * soma_x2) - (soma_x ** 2)
    termo_variancia_y = (n * soma_y2) - (soma_y ** 2)

    # Proteção contra variância nula ou imprecisão numérica negativa próxima de zero
    if termo_variancia_x <= 1e-12 or termo_variancia_y <= 1e-12:
        return 0.0, "Variância nula (constante)"

    denominador = math.sqrt(termo_variancia_x * termo_variancia_y)
    if denominador == 0.0:
        return 0.0, "Denominador zero"

    r = numerador / denominador

    # Clamping de segurança para evitar ex: 1.0000000000000002 por ponto flutuante
    r = max(-1.0, min(1.0, r))

    return r, classificar_correlacao(r)


def classificar_correlacao(r: float) -> str:
    """Interpreta a intensidade e o sentido da correlação estatística."""
    abs_r = abs(r)
    sentido = "Positiva" if r > 0 else ("Negativa" if r < 0 else "Nula")

    if abs_r >= 0.70:
        intensidade = "Forte"
    elif abs_r >= 0.40:
        intensidade = "Moderada"
    elif abs_r >= 0.15:
        intensidade = "Fraca"
    else:
        intensidade = "Desprezível / Muito Fraca"

    return f"{intensidade} ({sentido})"


def reducer_final(
    chave: str, valores: Iterable[VetorSuficiente]
) -> Tuple[str, Dict[str, object]]:
    """
    ETAPA REDUCE (Executada nos nós Reducers):
    Recebe todas as somas parciais associadas a uma coluna Y, consolida os
    somatórios globais e aplica a equação de correlação de Pearson.
    """
    _, (n, soma_x, soma_y, soma_x2, soma_y2, soma_xy) = combiner_parcial(chave, valores)

    if chave == "__AUDITORIA_DESCARTADAS__":
        return chave, {"linhas_descartadas": n}

    r, interpretacao = calcular_pearson_seguro(n, soma_x, soma_y, soma_x2, soma_y2, soma_xy)

    media_x = soma_x / n if n > 0 else 0.0
    media_y = soma_y / n if n > 0 else 0.0

    resultado = {
        "coluna_base_x": NOME_COLUNA_ALVO,
        "coluna_comparada_y": chave,
        "amostras_validas_n": n,
        "correlacao_pearson_r": round(r, 6),
        "interpretacao": interpretacao,
        "media_temp_x": round(media_x, 4),
        "media_coluna_y": round(media_y, 4),
        "somatorios": {
            "sum_x": round(soma_x, 4),
            "sum_y": round(soma_y, 4),
            "sum_x2": round(soma_x2, 4),
            "sum_y2": round(soma_y2, 4),
            "sum_xy": round(soma_xy, 4),
        },
    }
    return chave, resultado


# ==============================================================================
# 4. MODOS DE EXECUÇÃO: HADOOP STREAMING (STDIN/STDOUT) E EXECUÇÃO COMPLETA
# ==============================================================================

def executar_modo_mapper_streaming() -> None:
    """
    Modo Hadoop Streaming (--mapper):
    Lê linhas de sys.stdin e emite 'CHAVE\tJSON_VETOR' em sys.stdout.
    """
    for linha in sys.stdin:
        for chave, vetor in mapper_linha(linha):
            print(f"{chave}\t{json.dumps(vetor)}")


def executar_modo_combiner_streaming() -> None:
    """
    Modo Hadoop Streaming (--combiner):
    Agrega registros ordenados por chave em sys.stdin e emite somas parciais.
    """
    chave_atual: Optional[str] = None
    buffer_valores: List[VetorSuficiente] = []

    for linha in sys.stdin:
        linha = linha.strip()
        if not linha or "\t" not in linha:
            continue
        chave, valor_str = linha.split("\t", 1)
        vetor = tuple(json.loads(valor_str))

        if chave_atual is not None and chave != chave_atual:
            _, soma_parcial = combiner_parcial(chave_atual, buffer_valores)
            print(f"{chave_atual}\t{json.dumps(soma_parcial)}")
            buffer_valores = []

        chave_atual = chave
        buffer_valores.append(vetor)  # type: ignore[arg-type]

    if chave_atual is not None and buffer_valores:
        _, soma_parcial = combiner_parcial(chave_atual, buffer_valores)
        print(f"{chave_atual}\t{json.dumps(soma_parcial)}")


def executar_modo_reducer_streaming() -> None:
    """
    Modo Hadoop Streaming (--reducer):
    Lê os pares ordenados vindos do Shuffle/Sort em sys.stdin, executa o Reducer
    e imprime o resultado final de correlação de cada coluna.
    """
    chave_atual: Optional[str] = None
    buffer_valores: List[VetorSuficiente] = []

    for linha in sys.stdin:
        linha = linha.strip()
        if not linha or "\t" not in linha:
            continue
        chave, valor_str = linha.split("\t", 1)
        vetor = tuple(json.loads(valor_str))

        if chave_atual is not None and chave != chave_atual:
            _, resultado = reducer_final(chave_atual, buffer_valores)
            print(f"{chave_atual}\t{json.dumps(resultado, ensure_ascii=False)}")
            buffer_valores = []

        chave_atual = chave
        buffer_valores.append(vetor)  # type: ignore[arg-type]

    if chave_atual is not None and buffer_valores:
        _, resultado = reducer_final(chave_atual, buffer_valores)
        print(f"{chave_atual}\t{json.dumps(resultado, ensure_ascii=False)}")


def executar_pipeline_mapreduce_local(caminho_arquivo: str) -> Dict[str, Dict[str, object]]:
    """
    Simula fielmente todas as fases do ecossistema Hadoop MapReduce:
    1. Input Split & Map (com validação de integridade linha a linha)
    2. Combiner Local por bloco de dados
    3. Shuffle & Sort (agrupamento e ordenação por chave)
    4. Reduce (cálculo final da Correlação de Pearson)
    """
    tamanho_bloco_split = 150  # Simula múltiplos InputSplits distribuídos em Mappers distintos
    bloco_atual: List[Tuple[str, VetorSuficiente]] = []
    saida_combiners: List[Tuple[str, VetorSuficiente]] = []

    total_linhas_lidas = 0
    cabecalhos_ignorados = 0
    motivos_rejeicao: Dict[str, int] = defaultdict(int)

    with open(caminho_arquivo, mode="r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            total_linhas_lidas += 1
            integra, _, motivo = validar_e_extrair_linha(linha)
            if not integra:
                if motivo == "CABECALHO_CSV":
                    cabecalhos_ignorados += 1
                else:
                    motivos_rejeicao[motivo] += 1

            # Fase MAP
            for par_kv in mapper_linha(linha):
                bloco_atual.append(par_kv)

            # Quando o bloco atinge o limite do InputSplit, executa o COMBINER local do nó
            if total_linhas_lidas % tamanho_bloco_split == 0 and bloco_atual:
                agrupado_local: Dict[str, List[VetorSuficiente]] = defaultdict(list)
                for k, v in bloco_atual:
                    agrupado_local[k].append(v)
                for k, lista_v in agrupado_local.items():
                    saida_combiners.append(combiner_parcial(k, lista_v))
                bloco_atual = []

    # Processa o último bloco restante no Combiner
    if bloco_atual:
        agrupado_local = defaultdict(list)
        for k, v in bloco_atual:
            agrupado_local[k].append(v)
        for k, lista_v in agrupado_local.items():
            saida_combiners.append(combiner_parcial(k, lista_v))

    # Fase SHUFFLE & SORT: Agrupa por chave e ordena alfabeticamente como no Hadoop
    shuffle_agrupado: Dict[str, List[VetorSuficiente]] = defaultdict(list)
    for k, v in saida_combiners:
        shuffle_agrupado[k].append(v)

    # Fase REDUCE
    resultados_finais: Dict[str, Dict[str, object]] = {}
    for chave_ordenada in sorted(shuffle_agrupado.keys()):
        chave, res = reducer_final(chave_ordenada, shuffle_agrupado[chave_ordenada])
        resultados_finais[chave] = res

    # Exibe o Relatório Formatado de Execução
    exibir_relatorio_formatado(
        caminho_arquivo=caminho_arquivo,
        total_linhas=total_linhas_lidas,
        cabecalhos=cabecalhos_ignorados,
        motivos_rejeicao=dict(motivos_rejeicao),
        resultados=resultados_finais,
    )
    return resultados_finais


def exibir_relatorio_formatado(
    caminho_arquivo: str,
    total_linhas: int,
    cabecalhos: int,
    motivos_rejeicao: Dict[str, int],
    resultados: Dict[str, Dict[str, object]],
) -> None:
    """Imprime no terminal um painel executivo claro em Português (pt-BR)."""
    linhas_descartadas = sum(motivos_rejeicao.values())
    linhas_validas = total_linhas - cabecalhos - linhas_descartadas

    print("=" * 98)
    print(" RELATÓRIO DE EXECUÇÃO MAPREDUCE — CORRELAÇÃO COM A COLUNA 9 (TEMPERATURA / temp)")
    print("=" * 98)
    print(f" Arquivo Analisado          : {caminho_arquivo}")
    print(f" Total de Linhas Brutas     : {total_linhas}")
    print(f" Cabeçalhos Ignorados       : {cabecalhos}")
    print(f" Registros Íntegros (N)     : {linhas_validas}")
    print(f" Registros Corrompidos/Nulos: {linhas_descartadas}")
    if motivos_rejeicao:
        print(f" Detalhamento de Rejeições  : {json.dumps(motivos_rejeicao, ensure_ascii=False)}")
    print("-" * 98)
    print(f" {'COLUNA COMPARADA COM TEMPERATURA (COLUNA 9)':<54} | {'N':>4} | {'PEARSON (r)':>12} | {'INTERPRETAÇÃO'}")
    print("-" * 98)

    for chave, dados in resultados.items():
        if chave == "__AUDITORIA_DESCARTADAS__":
            continue
        r_val = float(dados["correlacao_pearson_r"])  # type: ignore[arg-type]
        n_val = int(dados["amostras_validas_n"])      # type: ignore[arg-type]
        interp = str(dados["interpretacao"])
        sinal = "+" if r_val >= 0 else ""
        print(f" {chave:<54} | {n_val:>4} | {sinal}{r_val:>11.6f} | {interp}")

    print("=" * 98)


# ==============================================================================
# 5. PONTO DE ENTRADA (CLI)
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Job MapReduce para cálculo da Correlação de Pearson entre a "
            "Coluna 9 (Temperatura) e as demais colunas numéricas do dataset Forest Fires."
        )
    )
    parser.add_argument(
        "arquivo",
        nargs="?",
        default="forestfires.csv",
        help="Caminho para o dataset CSV (padrão: forestfires.csv)",
    )
    parser.add_argument(
        "--mapper",
        action="store_true",
        help="Executa apenas a etapa Mapper lendo de STDIN (modo Hadoop Streaming)",
    )
    parser.add_argument(
        "--combiner",
        action="store_true",
        help="Executa apenas a etapa Combiner lendo de STDIN (modo Hadoop Streaming)",
    )
    parser.add_argument(
        "--reducer",
        action="store_true",
        help="Executa apenas a etapa Reducer lendo de STDIN (modo Hadoop Streaming)",
    )

    args = parser.parse_args()

    if args.mapper:
        executar_modo_mapper_streaming()
    elif args.combiner:
        executar_modo_combiner_streaming()
    elif args.reducer:
        executar_modo_reducer_streaming()
    else:
        executar_pipeline_mapreduce_local(args.arquivo)


if __name__ == "__main__":
    main()
