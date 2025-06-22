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
def collate_fn(batch):
        # Separa entradas e saídas em listas distintas
        entradas, saidas = zip(*batch)

        # Faz padding para que todas as entradas do batch tenham o mesmo tamanho
        entradas_padded = pad_sequence(entradas, batch_first=True, padding_value=0)

        # Faz o mesmo padding para as saídas
        saidas_padded = pad_sequence(saidas, batch_first=True, padding_value=0)

        # Retorna os tensores empacotados e alinhados por padding
        return entradas_padded, saidas_padded
