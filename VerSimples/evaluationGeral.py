import os
import json
import random
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from evaluation import CopyDatasetFromJSON
from NTM import NTM

def generate_temporary_dataset(path="dataset_avaliacao.json", num_samples=1000, max_len=10, vec_size=8):
    SEED = random.randint(0, 2**32 - 1)
    random.seed(SEED)
    torch.manual_seed(SEED)

    data = []
    for _ in range(num_samples):
        seq_len = random.randint(1, max_len)
        seq = torch.randint(0, 2, (seq_len, vec_size)).float()
        eos = torch.zeros(1, vec_size)
        eos[0, -1] = 1.0
        input_seq = torch.cat([seq, eos], dim=0)
        target_seq = seq.clone()
        data.append({"input": input_seq.tolist(), "target": target_seq.tolist()})

    with open(path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

    print(f"Novo dataset gerado com {num_samples} exemplos e salvo em '{path}' (SEED={SEED})")

def evaluate_metrics(model, dataset_path, num_samples=100, threshold=0.5, output_path="avaliacao_resultados.txt", examples_path="avaliacao_exemplos.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = CopyDatasetFromJSON(dataset_path)
    indices = random.sample(range(len(dataset)), num_samples)

    y_true_all = []
    y_pred_all = []

    with open(examples_path, "w", encoding="utf-8") as f_ex:
        for idx in indices:
            x, y_true = dataset[idx]
            x = x.unsqueeze(0).to(device)
            y_true = y_true.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(x)
                if out.size(1) > y_true.size(1):
                    out = out[:, :-1, :]
                y_pred = (out > threshold).float()

            y_true_all.append(y_true.cpu().numpy().reshape(-1))
            y_pred_all.append(y_pred.cpu().numpy().reshape(-1))

            f_ex.write(f"Exemplo {idx}\n")
            f_ex.write("Entrada (sem EOS):\n")
            f_ex.write(str(x[0, :-1].cpu().numpy()) + "\n")
            f_ex.write("Saída esperada:\n")
            f_ex.write(str(y_true[0].cpu().numpy()) + "\n")
            f_ex.write("Saída prevista:\n")
            f_ex.write(str(y_pred[0].cpu().numpy()) + "\n")
            f_ex.write("\n")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    f1 = f1_score(y_true_all, y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all)
    rec = recall_score(y_true_all, y_pred_all)

    results = (
        f"Avaliação em {num_samples} exemplos aleatórios:\n"
        f"F1 Score: {f1:.4f}\n"
        f"Acurácia (bit a bit): {acc:.4f}\n"
        f"Precisão: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
    )

    print("\n" + results)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(results)

if __name__ == "__main__":
    dataset_file = "dataset_avaliacao.json"
    generate_temporary_dataset(path=dataset_file, num_samples=1000,max_len=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NTM(input_size=8, output_size=8)
    model.load_state_dict(torch.load("checkpoint/best_model.pth", map_location=device))

    evaluate_metrics(
        model=model,
        dataset_path=dataset_file,
        num_samples=500,
        threshold=0.5,
        output_path="avaliacao_resultados.txt",
        examples_path="avaliacao_exemplos.txt"
    )


    dataset = CopyDatasetFromJSON("dataset_avaliacao.json")
    lengths = [x.shape[0] for x, _ in dataset]
    print(f"Comprimentos únicos: {sorted(set(lengths))}")