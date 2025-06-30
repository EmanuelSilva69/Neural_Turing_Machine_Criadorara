import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
import time

# Semente fixa para reprodutibilidade
SEED = int(time.time()) % (2**32)
random.seed(SEED)
torch.manual_seed(SEED)

class InversaoBitsDataset(Dataset):
    """
    Dataset para tarefa de inversão de bits. Gera pares (entrada, saída), onde:
    - entrada: sequência binária + marcador de fim (EOS)
    - saída: sequência binária com bits invertidos
    """
    def __init__(self, num_samples=10000, min_len=1, max_len=20, vector_size=8):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vector_size = vector_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_len = random.randint(self.min_len, self.max_len)
        seq = torch.randint(0, 2, (seq_len, self.vector_size)).float()
        target = 1.0 - seq  # Inversão de bits
        eos = torch.zeros(1, self.vector_size)
        eos[0, -1] = 1.0  # marcador EOS

        input_seq = torch.cat([seq, eos], dim=0)
        return input_seq, target

if __name__ == "__main__":
    dataset = InversaoBitsDataset(num_samples=15000, max_len=20, vector_size=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    data_json = []
    for i, (x, y) in enumerate(dataloader):
        entrada = x[0].tolist()
        saida = y[0].tolist()
        data_json.append({
            "input": entrada,
            "target": saida
        })

    filename = "dataset_inversao_bits.json"
    with open(filename, "w") as f:
        for exemplo in data_json:
            f.write(json.dumps(exemplo, separators=(",", ":")) + "\n")

    print(f"✅ Dataset salvo como '{filename}' com SEED = {SEED}")