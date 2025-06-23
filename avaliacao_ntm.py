import torch
import json
from NTM import NTM
from Tokenização import TOKEN_DICT

# === CONFIGURAÇÃO ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(TOKEN_DICT)
vocab_inv = {v: k for k, v in TOKEN_DICT.items()}
halt_token = TOKEN_DICT['HALT']

# === CARREGA EXEMPLO DO DATASET ===
with open("dataset_tokenizado.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    exemplo = dataset[0]  # Pega o primeiro exemplo real
    entrada_tokens = exemplo["entrada_tokenizada"]
    saida_esperada = exemplo["saida_tokenizada"]

print("📥 Entrada original (tokens):", entrada_tokens)
print("🎯 Saída esperada (tokens):", saida_esperada)

# === PREPARA ENTRADA ===
entrada = torch.tensor(entrada_tokens, dtype=torch.long).unsqueeze(0).to(device)

# === CARREGA MODELO COMPATÍVEL ===
model = NTM(
    input_dim=64,
    output_dim=vocab_size,
    controller_dim=64,
    memory_units=20,
    memory_dim=64,
    heads=3,
    controller_type='feedforward'
).to(device)

model.load_state_dict(torch.load("melhor_modelo_ntm.pth", map_location=device))
model.eval()

# === GERAÇÃO ITERATIVA ===
from torch.nn.functional import one_hot

saida_ids = []
max_tokens = 100
input_seq = torch.tensor(entrada_tokens, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    model.eval()
    output = model(input_seq)  # [1, T, vocab]
    last_token_logits = output[:, -1, :]  # último token
    next_token = last_token_logits.argmax(dim=-1)  # [1]

    for _ in range(max_tokens):
        saida_ids.append(next_token.item())

        if next_token.item() == halt_token:
            print("⚑ HALT detectado.")
            break

        # Novo input = entrada + saída parcial gerada
        combined = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
        input_seq = combined

        output = model(input_seq)
        next_token = output[:, -1, :].argmax(dim=-1)

# === AVALIAÇÃO ===
acertos = sum(p == t for p, t in zip(saida_ids, saida_esperada))
total = len(saida_esperada)
acc = 100 * acertos / total

# === RESULTADOS ===
print("📤 Saída gerada (tokens):", saida_ids)
print("📝 Saída decodificada:", " ".join(str(vocab_inv.get(tok, '?')) for tok in saida_ids))
print(f"✅ Acurácia token a token: {acertos}/{total} ({acc:.2f}%)")
