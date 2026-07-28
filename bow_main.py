from models.bow import BowModel


def preprocess(text: str):
    text = text.lower()

    return text


model = BowModel()

while True:

    entrada = input("\nDigite um texto:\n> ")

    if entrada.lower() in {"sair", "exit", "quit"}:
        print("Encerrando...")
        break

    texto = preprocess(entrada)

    tokens = preprocess(entrada).split()

    resultado = model.predict(texto)
 

    print("\nO que o modelo vê:", tokens)
    print("Classe:", resultado["prediction"])

    print("\nProbabilidades:")

    for classe, prob in resultado["probabilities"].items():
        print(f"  {classe}: {prob:.3f}")