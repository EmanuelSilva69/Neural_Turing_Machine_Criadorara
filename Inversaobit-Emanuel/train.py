import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from NTM import NTM
import json

# Dataset carregado do JSON
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
def init_memory(n_locations=128, word_size=20, value=1e-6):
    return torch.full((n_locations, word_size), value)
# Função de colagem para DataLoader com padding, protegendo contra sequências vazias
def collate_fn(batch):
    x_seqs, y_seqs = zip(*batch)
    x_seqs = [x for x in x_seqs if x.size(0) > 0]
    y_seqs = [y for y in y_seqs if y.size(0) > 0]
    if not x_seqs:
        return torch.zeros(1, 1, 8), torch.zeros(1, 1, 8)
    x_padded = pad_sequence(x_seqs, batch_first=True)
    y_padded = pad_sequence(y_seqs, batch_first=True)
    return x_padded, y_padded

# Função de treinamento
def train_ntm_full():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    input_size = 8
    output_size = 8
    model = NTM(input_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # L2 regularization
    loss_fn = nn.BCELoss()

    # Carregar dataset do arquivo JSON
    full_dataset = CopyDatasetFromJSON("dataset_inversao_bits.json")

    # Dividir dataset
    train_len = int(0.7 * len(full_dataset))
    val_len = int(0.15 * len(full_dataset))
    test_len = len(full_dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=collate_fn)

    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_path = "checkpoint/best_model.pth"

    os.makedirs("checkpoint", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            if x.size(1) == 0:
                continue
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if out.size(1) > y.size(1):
                out = out[:, :-1, :]
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Validação
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                if x.size(1) == 0:
                    continue
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                if out.size(1) > y.size(1):
                    out = out[:, :-1, :]
                loss = loss_fn(out, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Salvar melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoint/best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> Melhor modelo salvo em '{best_model_path}'")

    # Carregar melhor modelo salvo
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nMelhor modelo carregado de '{best_model_path}'")

    # Avaliação final no conjunto de teste
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            if x.size(1) == 0:
                continue
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if out.size(1) > y.size(1):
                out = out[:, :-1, :]
            loss = loss_fn(out, y)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\nTeste Loss Final: {avg_test_loss:.4f}")

    # Plotagem
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model

# Para executar:
if __name__ == "__main__":
    trained_model = train_ntm_full()
    torch.save(trained_model.state_dict(), "ntm_trained_model.pth")
    print("Modelo treinado salvo como 'ntm_trained_model.pth'")