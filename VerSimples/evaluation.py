import torch
import numpy as np
import json

class CopyDatasetFromJSON(torch.utils.data.Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                x = torch.tensor(item["input"], dtype=torch.float32)
                y = torch.tensor(item["target"], dtype=torch.float32)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def evaluate_interactive(model, dataset_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CopyDatasetFromJSON(dataset_path)
    model = model.to(device)
    model.eval()

    def evaluate_tensor(input_tensor):
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            out = model(input_tensor)
            y_pred = (out[:, :-1, :] > threshold).float()
        return y_pred

    while True:
        user_input = input("Digite um índice do dataset, 'input' para digitar os dados, ou 'exit' para sair: ")

        if user_input.lower() == 'exit':
            print("Encerrando avaliação.")
            break

        elif user_input.lower() == 'input':
            try:
                seq_len = int(input("Digite o tamanho da sequência: "))
                vec_size = 8
                seq = []
                for i in range(seq_len):
                    vec = input(f"Vetor {i+1} (8 bits separados por espaço): ")
                    bits = list(map(float, vec.strip().split()))
                    if len(bits) != vec_size:
                        raise ValueError("Vetor com tamanho incorreto.")
                    seq.append(bits)

                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                eos = torch.zeros(1, vec_size)
                eos[0, -1] = 1.0
                input_tensor = torch.cat([seq_tensor, eos], dim=0).unsqueeze(0)

                pred = evaluate_tensor(input_tensor)
                print("\nEntrada:")
                print(input_tensor[0, :-1])
                print("\nSaída prevista:")
                print(pred[0])
            except Exception as e:
                print(f"Erro ao processar entrada personalizada: {e}")

        else:
            try:
                idx = int(user_input)
                if idx < 0 or idx >= len(dataset):
                    print(f"Índice fora do intervalo. Use um valor entre 0 e {len(dataset)-1}.")
                    continue

                x, y_true = dataset[idx]
                x = x.unsqueeze(0).to(device)
                y_true = y_true.unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(x)
                    if out.size(1) > y_true.size(1):
                        out = out[:, :-1, :]
                    y_pred = (out > threshold).float()

                print(f"\n Exemplo do dataset — índice {idx}")
                print("Entrada (sem EOS):")
                print(x[0, :-1].cpu())
                print("\nSaída esperada:")
                print(y_true[0].cpu())
                print("\nSaída prevista:")
                print(y_pred[0].cpu())

                acc = (y_pred == y_true).float().mean().item()
                print(f"\n Acurácia bit a bit: {acc:.2%}\n")

            except ValueError:
                print("Por favor, digite um número válido, 'input' ou 'exit'.")

if __name__ == "__main__":
    from NTM import NTM

    model = NTM(input_size=8, output_size=8)
    model.load_state_dict(torch.load("checkpoint/best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    evaluate_interactive(model, "dataset.json")
