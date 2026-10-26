#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
IMPLEMENTAÇÃO COM ESTRUTURA DE CLASSE MRJOB (MRCorrelacaoIncendios)
Compatível com a biblioteca `mrjob` no Hadoop/EMR/Local e dotada de camada de
compatibilidade nativa para Python 3.12+ / 3.14 caso `mrjob` não esteja instalado.
================================================================================
"""

from typing import Any, Dict, Generator, Iterable, Tuple
from correlacao_mapreduce import (
    mapper_linha,
    combiner_parcial,
    reducer_final,
    executar_pipeline_mapreduce_local,
    VetorSuficiente,
)

try:
    from mrjob.job import MRJob  # type: ignore[import-not-found]
    MRJOB_DISPONIVEL = True
except ImportError:
    MRJOB_DISPONIVEL = False

    class MRJob:  # type: ignore[no-redef]
        """
        Camada de compatibilidade (Shim) para ambientes Python 3.13/3.14 onde o
        pacote legado `mrjob` não está instalado, mantendo a mesma interface OO.
        """
        @classmethod
        def run(cls) -> None:
            import sys
            arquivo = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "forestfires.csv"
            executar_pipeline_mapreduce_local(arquivo)


class MRCorrelacaoIncendios(MRJob):
    """
    Classe MapReduce estruturada segundo o padrão oficial do framework MRJob.

    Fluxo nos Nós do Cluster Hadoop:
    1. mapper( _, linha )   -> Valida integridade da linha e emite (coluna_y, [1, x, y, x², y², xy])
    2. combiner( chave, v ) -> Pré-agrega os somatórios localmente no nó escravo (DataNode)
    3. reducer( chave, v )  -> Consolida os somatórios globais e calcula o r de Pearson
    """

    def mapper(
        self, _: Any, linha: str
    ) -> Generator[Tuple[str, VetorSuficiente], None, None]:
        """
        Executado nos nós Mappers (Slaves):
        Filtra cabeçalhos, linhas truncadas, valores nulos, malformatados ou
        fora do domínio físico antes de emitir os vetores para o Shuffle.
        """
        for chave, vetor_estatistico in mapper_linha(linha):
            yield chave, vetor_estatistico

    def combiner(
        self, chave: str, valores: Iterable[VetorSuficiente]
    ) -> Generator[Tuple[str, VetorSuficiente], None, None]:
        """
        Executado localmente após o Mapper para reduzir tráfego de rede no Shuffle.
        """
        chave_saida, soma_parcial = combiner_parcial(chave, valores)
        yield chave_saida, soma_parcial

    def reducer(
        self, chave: str, valores: Iterable[VetorSuficiente]
    ) -> Generator[Tuple[str, Dict[str, object]], None, None]:
        """
        Executado nos nós Reducers:
        Agrega os somatórios finais (N, Σx, Σy, Σx², Σy², Σxy) e calcula a
        Correlação Linear de Pearson com tratamento contra divisão por zero.
        """
        chave_saida, resultado_estatistico = reducer_final(chave, valores)
        if chave_saida != "__AUDITORIA_DESCARTADAS__":
            yield chave_saida, resultado_estatistico


if __name__ == "__main__":
    MRCorrelacaoIncendios.run()
