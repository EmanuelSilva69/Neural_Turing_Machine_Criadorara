import csv
import json
from itertools import product

# Gera fitas binárias de até N bits
def gerar_fitas_binarias(max_bits=5):
    fitas = []
    for n in range(1, max_bits + 1):
        for combinacao in product("01", repeat=n):
            fita = ''.join(combinacao)
            fitas.append(fita)
    return fitas

# Gera regras da MT para cópia
def gerar_regras_copia(fita: str):
    regras = []
    for i, bit in enumerate(fita):
        regras.append([f"q{i}", bit, f"q{i+1}", bit, "R"])
    regras.append([f"q{len(fita)}", "_", "HALT", "_", "S"])
    return regras

#  Geração
fitas = gerar_fitas_binarias(16)

# CSV: salva regras como texto flat (opcional)
with open("dataset_copy_mt.csv", mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["entrada", "regras"])  # cabeçalho
    for fita in fitas:
        entrada = f"[TASK=COPY] {fita}"
        regras = gerar_regras_copia(fita)
        regras_flat = "; ".join(" ".join(regra) for regra in regras)
        writer.writerow([entrada, regras_flat])

# JSON: regras estruturadas corretamente (lista de listas)
dataset = []
for fita in fitas:
    entrada = f"[TASK=COPY] {fita}"
    regras = gerar_regras_copia(fita)  # lista de listas
    dataset.append({
        "entrada": entrada,
        "regras": regras  # mantém como lista de 5-elementos por regra
    })

with open("dataset_copy_mt.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
