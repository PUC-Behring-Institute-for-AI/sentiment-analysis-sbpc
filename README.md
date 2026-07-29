# Sentiment Analysis em Português: da Bag of Words aos Transformers

Este projeto demonstra, de forma interativa, como diferentes modelos de IA realizam **Análise de Sentimentos (Sentiment Analysis)** em textos em português do Brasil.

O objetivo é mostrar como a representação do texto evolui desde modelos clássicos baseados em frequência de palavras até modelos modernos baseados em Transformers.

A interface principal foi desenvolvida em **Gradio**, permitindo comparar, lado a lado, como diferentes modelos analisam exatamente o mesmo texto.

---

# Objetivos

Este repositório foi desenvolvido para fins educacionais e de divulgação científica. O projeto busca responder perguntas como:

- Como um computador representa um texto?
- O que muda entre um modelo simples e um modelo moderno?
- Qual o impacto do aumento da complexidade do modelo na qualidade das previsões?

Além disso, o usuário pode fornecer qualquer texto em português e comparar as previsões produzidas por diferentes modelos.

---

# Modelos implementados

## 🟦 1. Bag of Words (BoW)

Modelo clássico baseado em frequência de palavras.

Características:

- Representação vetorial baseada na ocorrência das palavras;
- Implementado com `CountVectorizer`;
- Classificador `Multinomial Naive Bayes`;
- Treinado do zero utilizando o corpus de treinamento.

Durante a execução, o usuário pode visualizar:

- representação Bag of Words (somente palavras presentes no vocabulário);
- probabilidades previstas;
- classe final.

---

## 🟩 2. BERTimbau Base Fine-Tuned

Modelo baseado em Transformers pré-treinado para português.

Características:

- Modelo base: `neuralmind/bert-base-portuguese-cased`;
- Fine-tuning para classificação binária de sentimentos;
- Implementado utilizando Hugging Face Transformers.

Durante a execução são exibidos:

- tokens WordPiece;
- vetores (embeddings) dos tokens (primeiras dimensões);
- probabilidades previstas;
- classe final.

---

# Interface interativa

A demonstração principal é executada através do **Gradio**.

Execute:

```bash
python app.py
```

Será iniciado um servidor local semelhante a:

```text
Running on local URL:
http://127.0.0.1:7860
```

Basta abrir esse endereço no navegador para utilizar a interface.

A interface permite comparar, lado a lado:

| Bag of Words | BERTimbau |
|--------------|-----------|
| Classe prevista | Classe prevista |
| Probabilidades | Probabilidades |
| Representação Bag of Words | Tokens WordPiece |
| Funcionamento do modelo | Funcionamento do modelo |

---

# Estrutura do projeto

```text
dataset/
    dataset.csv

models/
    bow.py
    bertimbau.py

run_py/
    bow_main.py
    bertimbau_main.py

train/
    train_bow.py
    train_bertimbau.py

    saved_models/
        bow_pipeline.joblib

app.py
README.md
requirements.txt
```

---

# Dataset

O treinamento foi realizado utilizando uma versão concatenada do repositório **Brazilian Portuguese Sentiment Analysis Datasets**, que reúne diversos conjuntos públicos de análise de sentimentos em português.

**Referência**

https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

---

# Treinamento

## Bag of Words

O modelo BoW é treinado utilizando:

- CountVectorizer;
- Multinomial Naive Bayes;
- Balanceamento das classes por undersampling.

Execute:

```bash
python train/train_bow.py
```

O modelo treinado é salvo em:

```text
train/saved_models/bow_pipeline.joblib
```

---

## BERTimbau

O modelo BERTimbau é obtido por fine-tuning do modelo pré-treinado utilizando a biblioteca Hugging Face Transformers.

Execute:

```bash
python train/train_bertimbau.py
```

Os pesos **não são armazenados neste repositório**, pois possuem centenas de megabytes.

O modelo fine-tuned encontra-se publicado no Hugging Face:

**https://huggingface.co/jvomiranda/BERTimbau-Sent-Analysis**

Durante a execução do projeto, o modelo é baixado automaticamente a partir desse repositório.

---

# Executando individualmente os modelos

Além da interface Gradio, cada modelo pode ser executado individualmente pelo terminal.

## Bag of Words

```bash
python run_py/bow_main.py
```

O programa apresenta:

- representação Bag of Words;
- probabilidades;
- classificação.

---

## BERTimbau

```bash
python run_py/bertimbau_main.py
```

O programa apresenta:

- tokens WordPiece;
- vetores dos tokens (embeddings);
- probabilidades;
- classificação final.

---

# Bibliotecas utilizadas

- Python
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- pandas
- NumPy
- Gradio

---

# Próximos passos

Este projeto foi concebido para comparar diferentes níveis de complexidade em IA.

As próximas versões deverão incluir:

- TF-IDF;
- FastText;
- SBERT;
- LLMs para classificação de sentimentos;
- comparação simultânea entre todos os modelos.

---

# Disclaimer

Este projeto é disponibilizado exclusivamente para fins educacionais e de divulgação científica.
