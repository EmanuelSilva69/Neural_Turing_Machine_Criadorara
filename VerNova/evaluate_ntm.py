# evaluate_ntm.py

import torch
import matplotlib.pyplot as plt
import json
from ntm_architeture import NeuralTuringMachine
from Dataloader import vocab
from Gerador_dataset import generate_mt_for_sequence, inv_vocab

# Configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_SIZE = 64
WORD_SIZE = 32
CONTROLLER_SIZE = 100
NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1
PAD_IDX = vocab["<PAD>"]

# Modelo (mesmos parâmetros do treino)
ntm = NeuralTuringMachine(
    input_size=1,
    output_size=1,
    controller_size=CONTROLLER_SIZE,
    memory_size=MEMORY_SIZE,
    word_size=WORD_SIZE,
    num_read_heads=NUM_READ_HEADS,
    num_write_heads=NUM_WRITE_HEADS
).to(DEVICE)

# Carrega pesos treinados
ntm.load_state_dict(torch.load("ntm_model.pt", map_location=DEVICE))
ntm.eval()

def decode_sequence(output_tensor):
    return [int(round(x.item())) for x in output_tensor if int(round(x.item())) != PAD_IDX]

def plot_comparison(pred_seq, true_seq):
    plt.figure(figsize=(12, 3))
    plt.plot(pred_seq, label="Predicted", marker='o')
    plt.plot(true_seq, label="Target", marker='x')
    plt.title("Regras geradas vs. verdadeiras")
    plt.xlabel("Posição")
    plt.ylabel("Token")
    plt.legend()
    plt.grid()
    plt.show()

def per_token_error(pred_seq, true_seq):
    errors = [(float(abs(p - t))) for p, t in zip(pred_seq, true_seq)]
    for i, e in enumerate(errors):
        print(f"Token {i:2d}: erro absoluto = {e:.2f}")
    avg_error = sum(errors) / len(errors) if errors else 0.0
    print(f"Erro médio por token: {avg_error:.4f}")
    return avg_error

def detokenize_rules(token_seq):
    rules = []
    for i in range(0, len(token_seq), 5):
        if i + 4 < len(token_seq):
            rule = {
                "state": inv_vocab.get(token_seq[i], str(token_seq[i])),
                "read": inv_vocab.get(token_seq[i+1], str(token_seq[i+1])),
                "write": inv_vocab.get(token_seq[i+2], str(token_seq[i+2])),
                "move": inv_vocab.get(token_seq[i+3], str(token_seq[i+3])),
                "next": inv_vocab.get(token_seq[i+4], str(token_seq[i+4]))
            }
            rules.append(rule)
    return rules

def evaluate(sequence):
    with torch.no_grad():
        ntm.reset()
        input_tensor = torch.tensor(sequence, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        outputs = []
        for t in range(input_tensor.size(1)):
            out = ntm(input_tensor[:, t])
            outputs.append(out.squeeze())

        flat_out = torch.stack(outputs).view(-1)
        pred_tokens = decode_sequence(flat_out)
        target_rules = generate_mt_for_sequence(sequence)
        target_tokens = [token for rule in target_rules for token in rule]

        print("\nInput:", sequence)
        print("Predicted Tokens:", pred_tokens)
        print("Target Tokens:   ", target_tokens)

        plot_comparison(pred_tokens, target_tokens)
        avg_error = per_token_error(pred_tokens, target_tokens)

        print("\nDecodificação e Interpretação das Regras Geradas:")
        print(json.dumps(detokenize_rules(pred_tokens), indent=2))

        return avg_error

if __name__ == "__main__":
    entrada = [0, 1, 1, 0]
    evaluate(entrada)
