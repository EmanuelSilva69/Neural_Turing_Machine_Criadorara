import csv
from itertools import product

def gerar_fitas_binarias(max_bits=5):
    fitas = []
    for n in range(1, max_bits + 1):
        for combinacao in product("01", repeat=n):
            fita = ''.join(combinacao)
            fitas.append(fita)
    return fitas
#as fitas
fitas = gerar_fitas_binarias(10)
print(f"Total de pares gerados: {len(fitas)}")
print(fitas)
#gerador de regras para a cópia de fitas
def gerar_regras_copia(fita: str):
    regras = []
    for i, bit in enumerate(fita):
        estado_atual = f"q{i}"
        simbolo_lido = bit
        proximo_estado = f"q{i+1}"
        simbolo_escrito = bit
        direcao = "R"
        regras.append((estado_atual, simbolo_lido, proximo_estado, simbolo_escrito, direcao))
    
    # Regra final: parar ao encontrar branco
    regras.append((f"q{len(fita)}", "_", "HALT", "_", "S"))
    return regras


with open("dataset_copy_mt.csv", mode="w", newline='', encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerow(["entrada", "regras"])  # cabeçalho

    for fita in fitas:
        entrada = f"[TASK=COPY] {fita}"
        regras = gerar_regras_copia(fita)
        regras_texto = "; ".join([f"({q},{s})→({q2},{s2},{d})" for (q, s, q2, s2, d) in regras])
        writer.writerow([entrada, regras_texto])

# geração do json
import json

dataset = []

for fita in fitas:
    entrada = f"[TASK=COPY] {fita}"
    regras = gerar_regras_copia(fita)
    regras_texto = [f"({q},{s})→({q2},{s2},{d})" for (q, s, q2, s2, d) in regras]
    dataset.append({
        "entrada": entrada,
        "regras": regras_texto
    })

with open("dataset_copy_mt.json", "w") as f:
    json.dump(dataset, f, indent=2)