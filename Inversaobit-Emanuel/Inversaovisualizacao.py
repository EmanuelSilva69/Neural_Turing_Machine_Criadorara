import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from NTM import NTM

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from NTM import NTM

# Dataset adaptado para inversão de bits
class BitFlipDatasetFromJSON(torch.utils.data.Dataset):
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

def plot_example(input_seq, target_seq, predicted_seq, idx):
    fig, axs = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

    input_np = input_seq.cpu().numpy()
    target_np = target_seq.cpu().numpy()
    predicted_np = predicted_seq.cpu().numpy()
    erro_mask = (target_np != predicted_np).astype(float)

    axs[0].imshow(input_np, cmap="Greys", aspect="auto")
    axs[0].set_title("Entrada (sem EOS)")

    axs[1].imshow(target_np, cmap="Greens", aspect="auto")
    axs[1].set_title("Saída Esperada (bits invertidos)")

    axs[2].imshow(predicted_np, cmap="Blues", aspect="auto")
    axs[2].set_title("Saída Prevista")

    axs[3].imshow(erro_mask, cmap="Reds", aspect="auto")
    axs[3].set_title("Erros (bits incorretos em vermelho)")

    for ax in axs:
        ax.set_ylabel("Passo")
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"b{i}" for i in range(8)])
    axs[-1].set_xlabel("Bits")

    plt.tight_layout()
    plt.suptitle(f"Comparação Visual - Inversão de Bits - Exemplo {idx}", y=1.02, fontsize=14)
    plt.show()

def comparar_visualmente(model, dataset_path="dataset_inversao_bits.json", threshold=0.5, num_exemplos=3, mostrar_erro=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = BitFlipDatasetFromJSON(dataset_path)

    exemplos_corretos = []
    erro_plotado = False

    print(f"\n Procurando {num_exemplos} exemplos corretos aleatórios...")
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        x, y_true = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y_true = y_true.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            if out.size(1) > y_true.size(1):
                out = out[:, :-1, :]
            y_pred = (out > threshold).float()

        if not torch.equal(y_pred, y_true):
            if mostrar_erro and not erro_plotado:
                print(f"\n❌ Exemplo com erro detectado (índice {idx})")
                plot_example(x[0, :-1], y_true[0], y_pred[0], idx)
                erro_plotado = True
        else:
            if len(exemplos_corretos) < num_exemplos:
                exemplos_corretos.append((idx, x[0], y_true[0], y_pred[0]))
            if len(exemplos_corretos) >= num_exemplos and (not mostrar_erro or erro_plotado):
                break

    for idx, x0, y0, y_pred in exemplos_corretos:
        print(f"\n✔ Exemplo correto (índice {idx})")
        plot_example(x0[:-1], y0, y_pred, idx)

if __name__ == "__main__":
    model = NTM(input_size=8, output_size=8)
    model.load_state_dict(torch.load("checkpoint/best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    comparar_visualmente(model, dataset_path="dataset_inversao_bits.json", threshold=0.5, num_exemplos=5, mostrar_erro=True)
