import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
# Classe personalizada para carregar os exemplos de entrada/saída tokenizados
class MTDataset(Dataset):
    def __init__(self, json_path, max_len=None, subset_indices=None):
        # Carrega os pares tokenizados do arquivo
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Aplica filtro de comprimento, se necessário
        if max_len is not None:
            data = [
                ex for ex in data
                if len(ex["entrada_tokenizada"]) <= max_len and len(ex["saida_tokenizada"]) <= max_len
            ]

        # Aplica subset por índice, se fornecido
        if subset_indices is not None:
            data = [data[i] for i in subset_indices if i < len(data)]

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entrada = torch.tensor(self.data[idx]["entrada_tokenizada"], dtype=torch.long)
        saida = torch.tensor(self.data[idx]["saida_tokenizada"], dtype=torch.long)
        return entrada, saida

# Função auxiliar para formar batches com padding automático
from torch.nn.utils.rnn import pad_sequence
import torch
def collate_fn(batch):
    entradas, saidas = zip(*batch)
    entradas = [torch.tensor(seq, dtype=torch.long) for seq in entradas]
    saidas = [torch.tensor(seq, dtype=torch.long) for seq in saidas]

    PAD_IDX = 0  # novo índice para '<PAD>'
    entradas_padded = pad_sequence(entradas, batch_first=True, padding_value=PAD_IDX)
    saidas_padded = pad_sequence(saidas, batch_first=True, padding_value=PAD_IDX)

    return entradas_padded, saidas_padded