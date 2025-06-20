from torch.nn.utils.rnn import pad_sequence
import torch

# tokens para entrada e saída
TOKEN_DICT = {
    '[TASK=COPY]': 0,
    '0': 1,
    '1': 2,
    '_': 3,
    'R': 4,
    'L': 5,
    'S': 6,
    'HALT': 7,
    '→': 8,  # opcional
}

# estados são adicionados dinamicamente
for i in range(100):  # até 100 estados
    TOKEN_DICT[f'q{i}'] = len(TOKEN_DICT)
    #tokenizar a entrada
def tokenize_input(task: str, tape: str) -> list[int]:
    tokens = [TOKEN_DICT[f'[TASK={task.upper()}]']]
    for bit in tape:
        tokens.append(TOKEN_DICT[bit])
    return tokens
#teste
tokens = tokenize_input('COPY', '1010')
print(tokens)  # [0, 2, 1, 2, 1]
#função pra copiar sequencia
def generate_copy_rules(tape: str) -> list[list[str]]:
    rules = []
    for i, bit in enumerate(tape):
        rules.append([f'q{i}', bit, f'q{i+1}', bit, 'R'])
    rules.append([f'q{len(tape)}', '_', 'HALT', '_', 'S'])
    return rules

rules = generate_copy_rules('1010')
# [['q0','1','q1','1','R'], ['q1','0','q2','0','R'], ..., ['q4','_','HALT','_','S']]
print(rules)
# função para tokenizar a saída
def tokenize_output(rules: list[list[str]]) -> list[int]:
    return [TOKEN_DICT[token] for rule in rules for token in rule]
tokenized = tokenize_output(rules)
print(tokenized)
# [q0, 1, q1, 1, R, q1, 0, q2, 0, R, ..., q4, _, HALT, _, S]

import csv
import json

# função para processar uma linha do CSV
def processar_exemplo(entrada_texto: str, regras_texto: str):
    # exemplo: entrada_texto = "[TASK=COPY] 1010"
    task, fita = entrada_texto.strip().split()
    entrada_tokens = tokenize_input(task=task.strip("[TASK=]"), tape=fita)

    # exemplo: regras_texto = "(q0,1)→(q1,1,R); (q1,0)→(q2,0,R); ..."
    regras_raw = regras_texto.strip().split("; ")
    regras = []
    for regra in regras_raw:
        esquerda, direita = regra.split("→")
        q, s = esquerda.strip("()").split(",")
        q2, s2, d = direita.strip("()").split(",")
        regras.append([q, s, q2, s2, d])

    saida_tokens = tokenize_output(regras)
    return entrada_tokens, saida_tokens

# carregar e processar CSV
dataset_tokenizado = []
with open("dataset_copy_mt.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for linha in reader:
        entrada_txt = linha["entrada"]
        regras_txt = linha["regras"]
        entrada_tokens, saida_tokens = processar_exemplo(entrada_txt, regras_txt)
        dataset_tokenizado.append({
            "entrada_tokenizada": entrada_tokens,
            "saida_tokenizada": saida_tokens
        })

# salvar como .json
with open("dataset_tokenizado.json", "w", encoding="utf-8") as f:
    json.dump(dataset_tokenizado, f, indent=2)