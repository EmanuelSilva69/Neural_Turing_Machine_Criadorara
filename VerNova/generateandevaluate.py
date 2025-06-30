# generate_and_evaluate.py

import torch
from evaluate_ntm import evaluate
from Gerador_dataset import generate_all_combinations_for_length
import random

# Configurações
BIT_RANGE = range(6, 13)  # Testar de 6 a 12 bits
SAMPLES_PER_LEN = 5       # Quantos exemplos testar por comprimento
SHUFFLE = True

erro_por_bitlen = {}

for bit_len in BIT_RANGE:
    print(f"\n================== Avaliando entradas com {bit_len} bits ==================")
    all_inputs = generate_all_combinations_for_length(bit_len)
    if SHUFFLE:
        random.shuffle(all_inputs)
    selected_inputs = all_inputs[:SAMPLES_PER_LEN]

    total_erro = 0
    for i, seq in enumerate(selected_inputs):
        print(f"\n----- Teste {i+1}/{SAMPLES_PER_LEN} para {bit_len} bits -----")
        erro = evaluate(list(seq))  # evaluate() deve retornar erro médio
        total_erro += erro

    avg_erro = total_erro / SAMPLES_PER_LEN
    erro_por_bitlen[bit_len] = avg_erro

print("\n\n======= Erro médio por comprimento =======")
for bitlen, err in erro_por_bitlen.items():
    print(f"{bitlen} bits: erro médio = {err:.4f}")
