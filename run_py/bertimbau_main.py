from models.bertimbau import BertimbauModel


model = BertimbauModel()

while True:

    texto = input("\nDigite um texto:\n> ")

    if texto.lower() in {
        "sair",
        "exit",
        "quit"
    }:
        break

    resultado = model.predict(texto)

    print("\nTokens WordPiece:")

    print(resultado["tokens"])

    print("\nToken IDs:")

    print(resultado["token_ids"])

    print("\nClasse:")

    print(resultado["prediction"])

    print("\nProbabilidades:")

    for classe, prob in resultado["probabilities"].items():

        print(f"{classe}: {prob:.3f}")