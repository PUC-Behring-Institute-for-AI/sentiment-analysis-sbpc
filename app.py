import gradio as gr

from models.bow import BowModel
from models.bertimbau import BertimbauModel


# ======================================================
# Carrega modelos
# ======================================================

bow_model = BowModel()
bert_model = BertimbauModel()


# ======================================================
# Inferência
# ======================================================

def predict(text):

    text = text.strip()

    if len(text) == 0:
        return (
            "",
            {},
            "",
            "",
            {},
            ""
        )

    # -------------------------
    # BoW
    # -------------------------

    bow = bow_model.predict(text)

    bow_prediction = (
        "😊 Positivo"
        if bow["prediction"] == 1
        else "😞 Negativo"
    )

    bow_probs = {
        "Positivo": bow["probabilities"][1],
        "Negativo": bow["probabilities"][0]
    }

    if len(bow["representation"]) == 0:

        bow_representation = (
            "Nenhuma palavra do texto pertence ao vocabulário do modelo."
        )

    else:

        bow_representation = "\n".join(

            f"{word:<20} {count}"

            for word, count in bow["representation"].items()

        )

    # -------------------------
    # BERT
    # -------------------------

    bert = bert_model.predict(text)

    bert_prediction = (
        "😊 Positivo"
        if bert["prediction"] == 1
        else "😞 Negativo"
    )

    bert_probs = {
        "Positivo": bert["probabilities"][1],
        "Negativo": bert["probabilities"][0]
    }

    bert_repr = "\n".join(
        f"{item['token']:<15} "
        f"[{', '.join(f'{v:.3f}' for v in item['vector'])}, ...]"
        for item in bert["representation"]
    )

    return (

        bow_prediction,
        bow_probs,
        bow_representation,

        bert_prediction,
        bert_probs,
        bert_repr

    )


# ======================================================
# Interface
# ======================================================

with gr.Blocks(
    title="Sentiment Analysis Playground"
) as demo:

    gr.Markdown(
"""
# Sentiment Analysis Playground

Compare como diferentes modelos processam um mesmo texto.

Atualmente a demonstração possui:

- 🟦 Bag of Words + Naive Bayes
- 🟩 BERTimbau Fine-Tuned

Digite qualquer texto em português e compare como cada modelo o representa internamente antes de classificá-lo.
"""
    )

    textbox = gr.Textbox(

        label="Texto",

        placeholder="Digite um texto...",

        lines=4

    )

    button = gr.Button(

        "Classificar",

        variant="primary"

    )

    with gr.Row():

        # =====================================================
        # BoW
        # =====================================================

        with gr.Column():

            gr.Markdown("## 🟦 Bag of Words")

            bow_prediction = gr.Textbox(

                label="Classe",

                interactive=False

            )

            bow_probs = gr.Label(

                label="Probabilidades",

                num_top_classes=2

            )

            with gr.Accordion(

                "Como o modelo representa o texto",

                open=False

            ):

                bow_repr = gr.Textbox(

                    label="Representação BoW",

                    lines=14,

                    interactive=False

                )

            gr.Markdown("""

### Como esse modelo funciona

- Conta palavras
- Ignora contexto
- Ignora ordem das palavras
- Baseado em frequência de termos

""")

        # =====================================================
        # BERT
        # =====================================================

        with gr.Column():

            gr.Markdown("## 🟩 BERTimbau")

            bert_prediction = gr.Textbox(

                label="Classe",

                interactive=False

            )

            bert_probs = gr.Label(

                label="Probabilidades",

                num_top_classes=2

            )

            with gr.Accordion(

                "Como o modelo representa o texto",

                open=False

            ):

                bert_repr = gr.Textbox(

                    label="Tokens WordPiece",

                    lines=14,

                    interactive=False

                )

            gr.Markdown("""

### Como esse modelo funciona

- Usa Transformer
- Considera contexto
- Considera ordem das palavras
- Usa mecanismo de atenção

""")

    button.click(

        predict,

        textbox,

        [

            bow_prediction,
            bow_probs,
            bow_repr,

            bert_prediction,
            bert_probs,
            bert_repr

        ]

    )

    textbox.submit(

        predict,

        textbox,

        [

            bow_prediction,
            bow_probs,
            bow_repr,

            bert_prediction,
            bert_probs,
            bert_repr

        ]

    )


demo.launch()