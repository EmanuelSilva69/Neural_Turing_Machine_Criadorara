import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
# Classe personalizada para carregar os exemplos de entrada/saída tokenizados
class MTDataset(Dataset):
    def __init__(self, json_path):
        # Abre o arquivo JSON contendo uma lista de exemplos
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)  # Carrega os dados como lista de dicionários

    def __len__(self):
        # Retorna o número total de exemplos no dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Recupera o exemplo de índice idx
        # Converte a entrada e a saída tokenizada para tensores do PyTorch
        entrada = torch.tensor(self.data[idx]["entrada_tokenizada"], dtype=torch.long)
        saida = torch.tensor(self.data[idx]["saida_tokenizada"], dtype=torch.long)
        return entrada, saida  # Retorna um par (entrada, saída)

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
