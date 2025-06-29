
import torch
import json
from evaluate_ntm import decode_sequence, detokenize_rules
from Gerador_dataset import generate_mt_for_sequence
from ntm_architeture import NeuralTuringMachine
from Dataloader import vocab

# Configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ntm_model.pt"
PAD_IDX = vocab["<PAD>"]

# Modelo (mesmos parâmetros do treino)
ntm = NeuralTuringMachine(
    input_size=1,
    output_size=1,
    controller_size=100,
    memory_size=64,
    word_size=32,
    num_read_heads=1,
    num_write_heads=1
).to(DEVICE)
ntm.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
ntm.eval()

def evaluate_and_export(sequence):
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

        errors = [float(abs(p - t)) for p, t in zip(pred_tokens, target_tokens)]
        avg_error = sum(errors) / len(errors) if errors else 0.0

        return {
            "input": sequence,
            "pred_tokens": pred_tokens,
            "target_tokens": target_tokens,
            "avg_error": avg_error,
            "rules": detokenize_rules(pred_tokens)
        }

if __name__ == "__main__":
    import random
    from Gerador_dataset import generate_all_combinations_for_length

    BIT_LEN = 6
    N = 10
    all_inputs = generate_all_combinations_for_length(BIT_LEN)
    random.shuffle(all_inputs)
    selected = all_inputs[:N]

    results = []
    for seq in selected:
        result = evaluate_and_export(list(seq))
        results.append(result)

    with open("results_export.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n✅ Exportados {len(results)} resultados para 'results_export.jsonl'")
