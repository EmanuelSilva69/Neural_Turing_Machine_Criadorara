import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Dataloader import BinaryCopyDataset, collate_fn, vocab
from ntm_architeture import NeuralTuringMachine
from Gerador_dataset import vocab
import matplotlib.pyplot as plt

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "dataset.json"
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
MEMORY_SIZE = 64
WORD_SIZE = 32
CONTROLLER_SIZE = 100
NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1
PAD_IDX = vocab["<PAD>"]
EMBED_DIM = 32
MODEL_PATH = "ntm_model.pt"
VOCAB_SIZE = max(vocab.values()) + 1

# Dataset completo e divisão aleatória
full_dataset = BinaryCopyDataset(jsonl_path=DATA_PATH)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Embedding
embedding = nn.Embedding(VOCAB_SIZE, embedding_dim=EMBED_DIM).to(DEVICE)

# Modelo
ntm = NeuralTuringMachine(
    input_size=EMBED_DIM,
    output_size=VOCAB_SIZE,
    controller_size=CONTROLLER_SIZE,
    memory_size=MEMORY_SIZE,
    word_size=WORD_SIZE,
    num_read_heads=NUM_READ_HEADS,
    num_write_heads=NUM_WRITE_HEADS,
    controller_type="lstm"
).to(DEVICE)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

ntm.apply(init_weights)
optimizer = optim.Adam(list(ntm.parameters()) + list(embedding.parameters()), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def accuracy(pred_logits, target, pad_value):
    pred = pred_logits.argmax(dim=-1)
    mask = (target != pad_value)
    correct = (pred == target) & mask
    return correct.sum().float() / mask.sum().clamp(min=1.0)

def run_epoch(loader, train=False):
    if train:
        ntm.train()
    else:
        ntm.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["input"].long().to(DEVICE)
            y = batch["output"].long().to(DEVICE)

            if y.max() >= VOCAB_SIZE or y.min() < 0:
                raise ValueError(f"Valores inválidos em y: min={y.min().item()}, max={y.max().item()}, VOCAB_SIZE={VOCAB_SIZE}")

            if train:
                optimizer.zero_grad()

            ntm.reset(batch_size=x.size(0), device=DEVICE)
            outputs = []
            for t in range(y.size(1)):
                input_token = x[:, t] if t < x.size(1) else torch.zeros_like(x[:, 0])
                input_emb = embedding(input_token)
                logits, _ = ntm(input_emb)
                outputs.append(logits)

            outputs = torch.stack(outputs, dim=1)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), y.view(-1))
            acc = accuracy(outputs, y, pad_value=PAD_IDX)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)

def train():
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "ntm_state_dict": ntm.state_dict(),
                "embedding_state_dict": embedding.state_dict()
            }, MODEL_PATH)
            torch.save({
                "ntm_state_dict": ntm.state_dict(),
                "embedding_state_dict": embedding.state_dict()
            }, f"checkpoints/ntm_val_{val_loss:.4f}_epoch_{epoch+1}.pt")  # salva uma versão única
            print(f"✓ Modelo salvo em '{MODEL_PATH}' (melhor até agora)")

    # Plotagem após treinamento
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss por Época")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy por Época")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    print("\n=== Teste Final ===")
    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train()