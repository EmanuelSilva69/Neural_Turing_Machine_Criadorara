from torch.utils.data import random_split, DataLoader
from formatacaodataset import MTDataset, collate_fn
from NTM import NTM
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Tokenização import TOKEN_DICT
# Hiperparâmetros
batch_size = 128
num_epochs = 400
seq_len = 80 # escolha baseada na análise do histograma Valiidação por currículo seria [20,35,50], ou apenas 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset base (sem filtro de tamanho ainda)
full_dataset = MTDataset("dataset_tokenizado.json")

# Split train/val/test fixo
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len
train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

#  Definição do vocabulário
# Carrega token_dict da tokenização

vocab_size = len(TOKEN_DICT)
print("Tamanho do vocabulário:", vocab_size)

#  Inicializa o modelo
model = NTM(
    input_dim=128,
    output_dim=vocab_size,
    controller_dim=128,
    memory_units=64,
    memory_dim=128,
    heads=3,
    controller_type='lstm'  # ou 'lstm'
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularização
criterion = nn.CrossEntropyLoss()
losses = []
val_losses = []
test_losses = []
best_val_loss = float('inf')
best_model_state = None
# Dataset auxiliar para wrapping
class ListaDataset(torch.utils.data.Dataset):
    def __init__(self, lista):
        self.lista = lista
    def __len__(self):
        return len(self.lista)
    def __getitem__(self, idx):
        return self.lista[idx]

# Função para filtrar por comprimento máximo
def filtra_por_seq_len(original_dataset, indices, max_len):
    subset = [original_dataset[i] for i in indices]
    return [
        (entrada, saida)
        for entrada, saida in subset
        if len(entrada) <= max_len and len(saida) <= max_len
    ]
#teste
import json
import matplotlib.pyplot as plt

with open("dataset_tokenizado.json", "r", encoding="utf-8") as f:
    data = json.load(f)

entrada_lens = [len(ex["entrada_tokenizada"]) for ex in data]
saida_lens = [len(ex["saida_tokenizada"]) for ex in data]

plt.hist(entrada_lens, bins=30, alpha=0.6, label="entrada")
plt.hist(saida_lens, bins=30, alpha=0.6, label="saida")
plt.axvline(x=20, color='red', linestyle='--')
plt.axvline(x=35, color='orange', linestyle='--')
plt.axvline(x=50, color='green', linestyle='--')
plt.axvline(x=60, color='blue', linestyle='--')
plt.legend()
plt.title("Distribuição dos comprimentos das sequências")
plt.show()

#  Loop de Treinamento
for epoch in range(num_epochs):

    
    # Filtra os datasets
    train_data = filtra_por_seq_len(full_dataset, train_set.indices, seq_len)
    val_data = filtra_por_seq_len(full_dataset, val_set.indices, seq_len)
    test_data = filtra_por_seq_len(full_dataset, test_set.indices, seq_len)
    train_loader = DataLoader(ListaDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ListaDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ListaDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Treinamento
    model.train()
    total_loss = 0
    for entradas, saidas in train_loader:
        entradas, saidas = entradas.to(device), saidas.to(device)
        outputs = model(entradas)
        outputs = outputs.reshape(-1, outputs.size(-1))
        saidas = saidas.reshape(-1)
        min_len = min(outputs.size(0), saidas.size(0))
        loss = criterion(outputs[:min_len], saidas[:min_len])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Clipping de gradientes
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    #  Validação
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for entradas, saidas in val_loader:
            entradas, saidas = entradas.to(device), saidas.to(device)
            outputs = model(entradas).reshape(-1, vocab_size)
            saidas = saidas.reshape(-1)
            min_len = min(outputs.size(0), saidas.size(0))
            val_loss += criterion(outputs[:min_len], saidas[:min_len]).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
        #  Salva o melhor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "melhor_modelo_ntm.pth")
        print(" Novo melhor modelo salvo!")
    print(f" Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} |  Val Loss: {val_loss:.4f}")

#  Restaura o melhor modelo antes do teste
model.load_state_dict(torch.load("melhor_modelo_ntm.pth"))
#  Avaliação Final (Teste)
test_loss = 0
model.eval()
with torch.no_grad():
    for entradas, saidas in test_loader:
        entradas, saidas = entradas.to(device), saidas.to(device)
        outputs = model(entradas).reshape(-1, vocab_size)
        saidas = saidas.reshape(-1)
        min_len = min(outputs.size(0), saidas.size(0))
        test_loss += criterion(outputs[:min_len], saidas[:min_len]).item()
test_loss /= len(test_loader)
print(f" Test Loss Final: {test_loss:.4f}")

#  Plota curvas
plt.plot(losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Treinamento vs Validação")
plt.grid(True)
plt.tight_layout()
plt.show()
