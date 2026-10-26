#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SUÍTE DE AUDITORIA E VALIDAÇÃO DE INTEGRIDADE (PROVA REAL PARA DEFESA)
Verifica:
1. Equivalência matemática exata entre o cálculo distribuído (Single-Pass MapReduce)
   e a fórmula estatística clássica centrada na média (Two-Pass Pearson).
2. Resiliência do Mapper contra dados corrompidos, nulos, truncados, NaN/Inf
   e valores fora do domínio físico, garantindo que nenhuma linha suja contamine
   os nós escravos (Reducers).
================================================================================
"""

import os
import math
import tempfile
from correlacao_mapreduce import (
    executar_pipeline_mapreduce_local,
    COLUNAS_NUMERICAS_COMPARACAO,
    INDICE_COLUNA_ALVO,
)


def calcular_pearson_classico_duas_passagens(caminho_csv: str) -> dict[str, float]:
    """Calcula a correlação de Pearson pela fórmula clássica centrada na média."""
    with open(caminho_csv, "r", encoding="utf-8") as f:
        linhas = [l.strip().split(",") for l in f if l.strip()]

    dados = linhas[1:]  # pula o cabeçalho
    valores_x = [float(linha[INDICE_COLUNA_ALVO]) for linha in dados]
    media_x = sum(valores_x) / len(valores_x)

    correlacoes_referencia: dict[str, float] = {}
    for idx_y, nome_y in COLUNAS_NUMERICAS_COMPARACAO.items():
        valores_y = [float(linha[idx_y]) for linha in dados]
        media_y = sum(valores_y) / len(valores_y)

        num = sum((x - media_x) * (y - media_y) for x, y in zip(valores_x, valores_y))
        den_x = sum((x - media_x) ** 2 for x in valores_x)
        den_y = sum((y - media_y) ** 2 for y in valores_y)
        r = num / math.sqrt(den_x * den_y)
        correlacoes_referencia[nome_y] = round(r, 6)

    return correlacoes_referencia


def executar_testes() -> None:
    print("\n[TESTE 1] Verificando precisão matemática do MapReduce vs Fórmula Clássica...")
    resultados_mr = executar_pipeline_mapreduce_local("forestfires.csv")
    referencia = calcular_pearson_classico_duas_passagens("forestfires.csv")

    for nome_col, r_esperado in referencia.items():
        r_obtido = float(resultados_mr[nome_col]["correlacao_pearson_r"])  # type: ignore[arg-type]
        diferenca = abs(r_esperado - r_obtido)
        assert diferenca < 1e-6, f"Divergência em {nome_col}: {r_obtido} != {r_esperado}"
    print(" -> SUCESSO: 100% das 10 correlações coincidem exatamente até a 6ª casa decimal!\n")

    print("[TESTE 2] Simulando injeção de falhas e linhas corrompidas no dataset...")
    linhas_sujas = [
        "",                                                       # Linha vazia
        "7,5,mar,fri,86.2,26.2,94.3",                             # Linha truncada (7 colunas != 13)
        "7,5,mar,fri,86.2,26.2,94.3,5.1,8.2,51,6.7,0,0,EXTRA",    # Coluna excedente (14 colunas)
        "7,5,mar,fri,86.2,26.2,94.3,5.1,NULL,51,6.7,0,0",         # Valor nulo na Temperatura (Col 9)
        "7,5,mar,fri,86.2,26.2,94.3,5.1,20.5,NaN,6.7,0,0",        # NaN na Umidade Relativa (Col 10)
        "7,5,mar,fri,86.2,26.2,94.3,5.1,ERRO_TXT,51,6.7,0,0",     # Texto malformatado na Col 9
        "7,5,mes_invalido,fri,86.2,26.2,94.3,5.1,18.0,51,6.7,0,0",# Mês inválido
        "7,5,mar,fri,86.2,26.2,94.3,5.1,999.9,51,6.7,0,0",        # Temperatura impossível (999.9 °C)
        "7,5,mar,fri,86.2,26.2,94.3,5.1,18.0,185,6.7,0,0",        # Umidade Relativa > 100% (185%)
    ]

    with open("forestfires.csv", "r", encoding="utf-8") as f_orig:
        conteudo_original = f_orig.read()

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(conteudo_original)
        for linha_ruim in linhas_sujas:
            tmp.write(linha_ruim + "\n")
        caminho_tmp = tmp.name

    try:
        resultados_sujos = executar_pipeline_mapreduce_local(caminho_tmp)
        descartadas = int(resultados_sujos["__AUDITORIA_DESCARTADAS__"]["linhas_descartadas"])  # type: ignore[arg-type]
        assert descartadas == len(linhas_sujas), f"Esperava {len(linhas_sujas)} linhas descartadas, obteve {descartadas}"

        for nome_col, r_esperado in referencia.items():
            r_apos_limpeza = float(resultados_sujos[nome_col]["correlacao_pearson_r"])  # type: ignore[arg-type]
            n_apos_limpeza = int(resultados_sujos[nome_col]["amostras_validas_n"])      # type: ignore[arg-type]
            assert n_apos_limpeza == 517, f"N deveria permanecer 517, mas foi {n_apos_limpeza}"
            assert abs(r_apos_limpeza - r_esperado) < 1e-6, "Correlação foi contaminada por linha suja!"
        print(" -> SUCESSO: Todas as anomalias foram barradas no Mapper e os cálculos nos Reducers permaneceram 100% íntegros!\n")
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)


if __name__ == "__main__":
    executar_testes()
