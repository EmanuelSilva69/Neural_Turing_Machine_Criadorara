import json

# Dicionário de tokens fixos
TOKEN_DICT = {
    '<PAD>': 0,
    '[TASK=COPY]': 1,
    '0': 2,
    '1': 3,
    '_': 4,
    'R': 5,
    'L': 6,
    'S': 7,
    'HALT': 8
}

# Estados dinâmicos q0, q1, ...
for i in range(100):
    TOKEN_DICT[f'q{i}'] = len(TOKEN_DICT)
# Tamanho do vocabulário
vocab_size = len(TOKEN_DICT)
print(" Tamanho do vocabulário:", vocab_size)
# Função para tokenizar a entrada
def tokenize_input(task: str, tape: str) -> list[int]:
    tokens = [TOKEN_DICT[f'[TASK={task.upper()}]']]
    for bit in tape:
        tokens.append(TOKEN_DICT[bit])
    return tokens

# Função para tokenizar a saída (token a token, sem ignorar)
def tokenize_output(regras_flat: list[str]) -> list[int]:
    tokens = []
    for token in regras_flat:
        if token not in TOKEN_DICT:
            raise KeyError(f"Token desconhecido: {token}")
        tokens.append(TOKEN_DICT[token])
    return tokens

# Tokenizar todo o dataset_copy_mt.json
with open("dataset_copy_mt.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

tokenizado = []
for idx, exemplo in enumerate(dataset):
    entrada_texto = exemplo["entrada"]
    regras_raw = exemplo["regras"]  # lista de listas
    
    # Achata as regras
    regras_flat = [token for regra in regras_raw for token in regra]
    
    task, fita = entrada_texto.split()
    entrada_tokens = tokenize_input(task.strip("[TASK=]"), fita)
    saida_tokens = tokenize_output(regras_flat)

    # Mostra o primeiro exemplo
    if idx == 0:
        print(" Entrada original:", entrada_texto)
        print(" Entrada tokenizada:", entrada_tokens)
        print(" Regras originais:", regras_raw)
        print(" Regras tokenizadas:", saida_tokens)
        print("-" * 50)

    tokenizado.append({
        "entrada_tokenizada": entrada_tokens,
        "saida_tokenizada": saida_tokens
    })

# Salvar como JSON final
with open("dataset_tokenizado.json", "w", encoding="utf-8") as f:
    json.dump(tokenizado, f, indent=2, ensure_ascii=False)

print(" Tokenização concluída. Exemplos salvos em dataset_tokenizado.json")
