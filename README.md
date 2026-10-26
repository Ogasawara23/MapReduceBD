# 🔥 Correlação Estatística de Pearson via MapReduce — Dataset Forest Fires

Projeto acadêmico de **Big Data & Ecossistema Hadoop** que implementa um job **MapReduce** em Python para calcular a **Correlação Linear de Pearson ($r$)** entre a **Coluna 9 (`temp` — Temperatura em °C)** e todas as demais 10 colunas numéricas do dataset *Forest Fires* (UCI Machine Learning Repository), com validação atômica de integridade de dados nos nós escravos (*Mappers/Combiners/Reducers*).

---

## 📁 Estrutura do Projeto

```text
Bigdata/
├── forestfires.csv          # Dataset oficial UCI Forest Fires (518 linhas: 1 cabeçalho + 517 registros)
├── correlacao_mapreduce.py  # Script principal MapReduce (Mapper, Combiner, Reducer + Validação Atômica)
├── correlacao_mrjob.py      # Classe MRCorrelacaoIncendios estruturada no padrão MRJob
├── testar_integridade.py    # Suíte de testes de precisão matemática e injeção de falhas/dados sujos
└── README.md                # Instruções rápidas de instalação e uso
```

---

## 🚀 Comandos Principais de Execução

### 1. Executar o Job MapReduce Completo (Modo Local com Relatório Formatado)
```bash
python3 correlacao_mapreduce.py forestfires.csv
```

### 2. Executar em Modo Hadoop Streaming Puro (Unix Pipes: `Map -> Sort -> Combine -> Sort -> Reduce`)
```bash
cat forestfires.csv \
  | python3 correlacao_mapreduce.py --mapper \
  | sort \
  | python3 correlacao_mapreduce.py --combiner \
  | sort \
  | python3 correlacao_mapreduce.py --reducer
```

### 3. Executar via Classe `MRCorrelacaoIncendios` (Padrão MRJob)
```bash
python3 correlacao_mrjob.py forestfires.csv
```

### 4. Executar a Auditoria de Integridade e Validação Matemática
```bash
python3 testar_integridade.py
```
