# tabformer-embedder

Embeddings de dados tabulares/transacionais com **duas arquiteturas**, API
estilo scikit-learn e **foco em máquinas com pouca memória** — não precisa de
GPU nem do pacote `transformers`.

## Arquiteturas

| | `arch="nuformer"` (padrão) | `arch="tabformer"` |
|---|---|---|
| Paper | [nuFormer, arXiv:2507.23267](https://arxiv.org/html/2507.23267v3) | TabFormer, arXiv:2011.01884 |
| Backbone | transformer **causal flat** (GPT-style) | hierárquico (campo → transação) |
| Pré-treino | **next-token prediction** | MLM (masked language modeling) |
| Positional embedding | **nenhum (NoPE)** | aprendido |
| Datas | decompostas: mês/dia/dia-da-semana/hora + ano em bins | bin único de timestamp |
| Negativos | token de **sign** + bin de magnitude | bin direto |
| Separador | `[SEP]` antes de cada transação | — |
| Representação do usuário | `pooling="last"` (último token) | mean/last/max |

O que **não** veio do paper do nuFormer (por escopo/hardware):
descrições em texto via BPE (só campos estruturados), o branch tabular
DCNv2 + PLR para features estáticas (fusão supervisionada), modelo de
24 camadas/330M, FlashAttention, LoRA e treino distribuído. O que veio é o
core: tokenização compacta por campo, causal + NoPE + NTP, datas decompostas,
sign+magnitude e representação pelo último token.

## Instalação

```bash
pip install -e .        # a partir da raiz do repositório TabFormer
```

Dependências: `torch>=2.0`, `numpy`, `pandas`.

## Uso

```python
import pandas as pd
from tabformer_embedder import TabFormerEmbedder

df = pd.read_csv("transacoes.csv")   # precisa ter uma coluna de usuário (ex.: "User")

emb = TabFormerEmbedder(
    user_col="User",
    target_col="Is Fraud?",          # opcional: coluna excluída dos embeddings
    arch="nuformer",                  # padrão; "tabformer" para o hierárquico
    datetime_cols=["Timestamp"],      # datas decompostas em mês/dia/weekday/hora
    categorical_cols=["MCC", "Zip"],  # códigos nominais: 1 token por valor
    seq_len=10,                       # janela de 10 transações
    stride=5,                         # janelas sobrepostas
    embedding_dim=128,
    epochs=5,
)

emb.fit(df)                              # pré-treino (NTP no modo nuformer)
E = emb.transform(df)                    # np.ndarray [n_janelas, 128]
E = emb.transform(df, pooling="last")    # representação do nuFormer (último token)
```

CSV direto (lido em blocos, sem carregar tudo na RAM):

```python
emb.fit("transacoes.csv", chunksize=200_000)
E = emb.transform("transacoes.csv", out_path="emb.npy")  # saída em memmap
```

Salvar / carregar:

```python
emb.save("tabformer.pt")
emb = TabFormerEmbedder.load("tabformer.pt")
```

## Como ele economiza memória

| Onde | O que faz |
|---|---|
| Pré-processamento | CSV lido em chunks (2 passadas); contadores de valores, não o dataset inteiro |
| Tokens | matriz `int32` em vez de listas Python (~15x menor); acima de ~256 MB vai para memmap em disco (`cache_dir`) |
| Modelo | pequeno por padrão: hidden 128, 2 camadas (~1–25M params); nuFormer flat é ainda mais leve que o hierárquico |
| Loss | logits só nas posições treinadas e em blocos de ~128 MB — vocabulário de 150k cabe em ~1,2 GB de RAM |
| Treino | `batch_size` pequeno + `grad_accum`; fp32 (CPU) ou AMP (CUDA) |
| Saída | `transform(..., out_path="emb.npy")` escreve direto em memmap de disco |

Parâmetros que ajudam quando a memória aperta:
- `min_freq=N` — categorias raras viram `[UNK]` e encolhem o vocabulário;
- `batch_size` menor + `grad_accum` maior (mesma batch efetiva, menos RAM);
- `embedding_dim` / `n_layers` menores;
- `cache_dir="./cache"` para os tokens irem ao disco.

## Referência rápida

- `fit(df_ou_csv, chunksize=...)` — ajusta vocabulário, pré-processa e pré-treina.
- `transform(data, pooling="mean"|"last"|"max"|"flatten", return_metadata=False, out_path=None)`
  — gera embeddings; com `return_metadata=True` devolve também um DataFrame com
  o usuário e a posição inicial de cada janela.
- `fit_transform(data)` — os dois em um passo.
- `embed(data)` — apelido de `transform`.
- `save(path)` / `TabFormerEmbedder.load(path, device="auto")`.

Usuários/categorias novos em `transform` são mapeados para `[UNK]` nos campos
(usuários novos ainda são agrupados corretamente em janelas próprias).

## Colunas: como cada tipo é tokenizado

| Tipo | Tokens |
|---|---|
| object/string | 1 categoria por valor (`min_freq` filtra raras) |
| int/float | quantile bins (`num_bins`) |
| int/float em `categorical_cols` | 1 categoria por valor (use p/ MCC, Zip, códigos) |
| negativos (se `sign_split`) | token sign (2) + bin da magnitude |
| datetime (se `decompose_dates`) | mês(≤12) + dia(≤31) + weekday(7) + hora(24) + ano em bins |
