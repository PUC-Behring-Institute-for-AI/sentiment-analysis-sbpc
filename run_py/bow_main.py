from models.bow import BowModel
import re

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def preprocess(text: str) -> str:
    text = text.lower()

    # O CountVectorizer considera apenas tokens com
    # duas ou mais letras/números e ignora pontuação.
    tokens = TOKEN_PATTERN.findall(text)

    return " ".join(tokens)

model = BowModel()

# Recupera o CountVectorizer treinado
vectorizer = model.pipeline.named_steps["vectorizer"]

while True:

    entrada = input("\nDigite um texto:\n> ")

    if entrada.lower() in {"sair", "exit", "quit"}:
        print("Encerrando...")
        break

    texto = preprocess(entrada)

    resultado = model.predict(texto)

    # Bag of Words
    X = vectorizer.transform([texto])

    vocab = vectorizer.get_feature_names_out()

    print("\n" + "=" * 70)
    print(" COMO O BAG OF WORDS ENXERGA O TEXTO")
    print("=" * 70)

    print("\nTexto original:")
    print(f"  {entrada}")

    print("\nApós preprocessamento:")
    print(f"  {texto}")

    print("\nTokens:")
    print(f"  {texto.split()}")

    print("\nRepresentação Bag of Words")
    print("(Apenas palavras presentes no texto)\n")

    if X.nnz == 0:
        print("Nenhuma palavra do texto pertence ao vocabulário aprendido.")
    else:
        for indice, frequencia in zip(X.indices, X.data):
            palavra = vocab[indice]
            print(
                f"Índice {indice:>6} | "
                f"{palavra:<25} | "
                f"Frequência = {int(frequencia)}"
            )

    if resultado["prediction"] is None:
        print("\nO modelo não conseguiu classificar o texto.")
        print("Isso pode ocorrer quando o texto não contém palavras do vocabulário aprendido.")
        print("=" * 70)
        continue
    elif resultado["prediction"] == 0.0:
        resultado_texto = "Negativo"
    elif resultado["prediction"] == 1.0:
        resultado_texto = "Positivo"
    print("\nPredição:")
    print(f"  Classe: {resultado_texto}")

    print("\nProbabilidades:")

    for classe, prob in resultado["probabilities"].items():
        print(f"  Classe {classe}: {prob:.2%}")

    print('Classe 0.0 = Negativo | Classe 1.0 = Positivo')

    print("=" * 70)