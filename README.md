# Sentiment Analysis em Português: da Bag of Words às LLMs

Este projeto demonstra, de forma interativa, como diferentes modelos de IA realizam **Análise de Sentimentos (Sentiment Analysis)** em textos em português do Brasil.

O objetivo é mostrar como a representação do texto evolui desde modelos clássicos baseados em frequência de palavras até modelos modernos baseados em Transformers.

## Objetivos

Este repositório foi desenvolvido para fins educacionais e de divulgação científica. O projeto busca responder perguntas como:

* Como um computador representa um texto?
* O que muda entre um modelo simples e um modelo moderno?
* Qual o impacto do aumento da complexidade do modelo na qualidade das previsões?

Além disso, o usuário pode fornecer qualquer texto em português e comparar as previsões produzidas por diferentes modelos.

## Modelos implementados até o momento

### 1. Bag of Words (BoW)

Modelo clássico baseado em frequência de palavras.

Características:

* Representação vetorial baseada na ocorrência das palavras.
* Implementado com `CountVectorizer`.
* Classificador `Multinomial Naive Bayes`.
* Treinado do zero utilizando o corpus de treinamento.

Durante a execução, o programa mostra:

* texto enviado;
* tokens reconhecidos;
* representação Bag of Words (apenas termos presentes);
* probabilidades previstas;
* classe final.

---

### 2. BERTimbau Base

Modelo baseado em Transformers pré-treinado para português.

Características:

* Modelo: `neuralmind/bert-base-portuguese-cased`
* Fine-tuning para classificação binária de sentimentos.
* Implementado com Hugging Face Transformers.

Durante a execução, serão exibidos:

* tokens WordPiece;
* IDs dos tokens;
* probabilidades previstas;
* classe final.

---


## Estrutura do projeto

```text

models/
    bow.py
    bertimbau.py

train/
    train_bow.py
    train_bertimbau.py

    /saved_models/
        bow_pipeline.joblib
        bertimbau/

bow_main.py  (executa o modelo BoW final treinado)
bert_main.py (executa o modelo BERTimbau final fine-tuned)
```
---

## Dataset

O treinamento foi realizado utilizando uma versão concatenada do repositório **Brazilian Portuguese Sentiment Analysis Datasets**, que reúne diversos conjuntos públicos de análise de sentimentos em português.

Referência:

http://kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets 


## Treinamento

### Bag of Words

O modelo BoW é treinado utilizando:

* CountVectorizer
* Multinomial Naive Bayes
* Balanceamento das classes por undersampling

```bash
python train/train_bow.py
```

O modelo treinado é salvo em:

```text
train/saved_models/bow_pipeline.joblib
```

---

### BERTimbau

O modelo BERTimbau é obtido por fine-tuning do modelo pré-treinado utilizando a biblioteca Hugging Face Transformers.

```bash
python train/train_bertimbau.py
```

Os pesos são salvos em:

```text
train/saved_models/bertimbau/
```

## Testando os modelos

### Bag of Words

```bash
python bow_main.py
```

O usuário pode digitar qualquer frase em português.

Exemplo:

```text
Esse filme foi maravilhoso!
```

O programa apresenta:

* representação Bag of Words;
* probabilidades;
* classificação.

---

### BERTimbau

```bash
python bert_main.py
```

O programa apresentará:

* tokenização WordPiece;
* IDs dos tokens;
* probabilidades;
* classificação final.

## Bibliotecas

* Python
* scikit-learn
* PyTorch
* Hugging Face Transformers
* pandas
* datasets
* NumPy

## Disclaimer

Este projeto é disponibilizado para fins educacionais e de divulgação científica.
