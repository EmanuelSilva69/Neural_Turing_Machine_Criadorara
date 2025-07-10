import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Importa os tokens definidos no GeradorDataset.py para garantir consistência
try:
    from GeradorDataset import TOKEN_DICT, PAD_TOKEN, START_TOKEN, END_TOKEN, TASK_COPY_TOKEN
except ImportError:
    print("Não foi possível importar tokens de GeradorDataset.py. Usando definição local.")
    # Fallback para definição local se a importação falhar.
    TOKEN_DICT = {
        '[PAD]': 0,
        '[INICIO]': 1,
        '[FIM]': 2,
        '[COPIA_TASK]': 3,
        '0': 4,
        '1': 5,
    }
    PAD_TOKEN = TOKEN_DICT['[PAD]']
    START_TOKEN = TOKEN_DICT['[INICIO]']
    END_TOKEN = TOKEN_DICT['[FIM]']
    TASK_COPY_TOKEN = TOKEN_DICT['[COPIA_TASK]']


class DatasetTarefaMaquinaTuring(Dataset):
    """
    Um Dataset personalizado para a tarefa de cópia da Neural Turing Machine.
    Carrega dados de um arquivo JSON.
    """
    def __init__(self, caminho_arquivo_json: str, tamanho_vetor_bit: int = 8):
        """
        Inicializa o dataset carregando os exemplos do arquivo JSON.
        
        Args:
            caminho_arquivo_json (str): O caminho para o arquivo JSON contendo os dados.
            tamanho_vetor_bit (int): A dimensão esperada para cada vetor de bit (ex: 8 para 8-bit random vectors).
        """
        with open(caminho_arquivo_json, 'r', encoding='utf-8') as f:
            self.exemplos = json.load(f)
        self.tamanho_vetor_bit = tamanho_vetor_bit

    def __len__(self):
        """Retorna o número total de exemplos no dataset."""
        return len(self.exemplos)

    def __getitem__(self, idx: int):
        """
        Retorna um exemplo de entrada e saída tokenizado e pré-processado.
        
        Args:
            idx (int): O índice do exemplo a ser recuperado.
            
        Returns:
            tuple: (entrada_numerica, saida_numerica)
                   entrada_numerica: Lista de vetores (tensores) para a sequência de entrada.
                   saida_numerica: Lista de vetores (tensores) para a sequência de saída.
        """
        exemplo = self.exemplos[idx]
        entrada_raw = exemplo['entrada']
        saida_raw = exemplo['saida']

        # Converte strings de bits e tokens especiais para vetores numéricos
        entrada_numerica = self._converter_sequencia_para_numerica(entrada_raw)
        saida_numerica = self._converter_sequencia_para_numerica(saida_raw)
        
        return entrada_numerica, saida_numerica

    def _converter_token_para_vetor(self, token: str):
        """
        Converte um token (string) em seu vetor numérico correspondente.
        
        Args:
            token (str): O token a ser convertido (ex: '01011001', '[FIM]').
            
        Returns:
            torch.Tensor: O vetor numérico correspondente ao token.
        """
        # Cria um vetor de zeros com o tamanho do vetor de bits + tamanho dos tokens especiais.
        # Por exemplo, se tamanho_vetor_bit=8 e há 4 tokens especiais, o vetor_representacao
        # terá 8 + 4 = 12 posições, com um hot-encoding para cada.
        
        # O tamanho do vocabulário total é o número de tokens especiais + 2 (para '0' e '1')
        # No entanto, os '0' e '1' são parte dos vetores de bits.
        # A representação do paper para os bits é o próprio vetor de bits (ex: [0,0,1,1,0,1,0,0]).
        # Para os tokens especiais, podemos usar uma representação one-hot ou um vetor de zeros/uns.
        # Vamos usar um vetor de tamanho (tamanho_vetor_bit) para os bits, e um vetor maior
        # para incluir os tokens especiais, onde os bits ocupam as primeiras posições.

        # Opção 1: Vetor de representação 'híbrido' onde bits são o vetor real e tokens especiais
        # são one-hot codificados em posições adicionais.
        
        # Dimensão final do vetor de entrada/saída de um passo de tempo para a NTM.
        # O paper original usa "8-bit random vectors" para o input, e os tokens delimitadores
        # são adicionados em canais separados. Vamos fazer uma representação onde o
        # tamanho de entrada da NTM é (tamanho_vetor_bit + num_tokens_especiais).
        
        # Contagem de tokens especiais para hot-encoding (exceto '0' e '1', que são bits)
        num_tokens_especiais_hot = len(TOKEN_DICT) - 2 # [PAD], [INICIO], [FIM], [COPIA_TASK]

        # O tamanho total da entrada/saída de um passo de tempo para a NTM
        # será o tamanho do vetor de bits + o espaço para os tokens especiais one-hot
        tamanho_total_passo_tempo = self.tamanho_vetor_bit + num_tokens_especiais_hot
        vetor_representacao = torch.zeros(tamanho_total_passo_tempo, dtype=torch.float32)

        if token in ['0', '1']: # Isso não deve acontecer com o GeradorDataset atual que mapeia '0'/'1' para parte de um vetor
            # Este bloco é para o caso de tokens '0' ou '1' serem passados individualmente.
            # No nosso GeradorDataset, os bits são parte de um vetor de 8 bits.
            # Se a entrada for um único bit, ajuste a lógica aqui.
            pass
        elif token in ['[PAD]', '[INICIO]', '[FIM]', '[COPIA_TASK]']:
            # Posição do hot-encoding para tokens especiais.
            # Mapeia '[PAD]' para índice 0 de tokens especiais, '[INICIO]' para 1, etc.
            # Esses índices vêm depois dos bits.
            if token == '[PAD]': idx_token = 0
            elif token == '[INICIO]': idx_token = 1
            elif token == '[FIM]': idx_token = 2
            elif token == '[COPIA_TASK]': idx_token = 3
            else: idx_token = -1 # Erro

            if idx_token != -1:
                vetor_representacao[self.tamanho_vetor_bit + idx_token] = 1.0
        else: # Assumimos que é uma string de bits (ex: '01011001')
            try:
                # Converte a string de bits para uma lista de floats (0.0 ou 1.0)
                bits_flutuantes = [float(bit) for bit in token]
                if len(bits_flutuantes) != self.tamanho_vetor_bit:
                    raise ValueError(f"Comprimento do vetor de bits '{token}' ({len(bits_flutuantes)}) não corresponde ao esperado ({self.tamanho_vetor_bit}).")
                vetor_representacao[:self.tamanho_vetor_bit] = torch.tensor(bits_flutuantes, dtype=torch.float32)
            except ValueError:
                raise ValueError(f"Token desconhecido ou formato de vetor de bits inválido: '{token}'")

        return vetor_representacao

    def _converter_sequencia_para_numerica(self, sequencia_tokens: list):
        """
        Converte uma lista de tokens (strings) em uma lista de tensores numéricos.
        """
        return [self._converter_token_para_vetor(token) for token in sequencia_tokens]


def collate_fn_pad_sequencias(batch, tamanho_vetor_bit: int = 8):
    """
    Função de colagem para DataLoader que aplica padding às sequências.
    
    Args:
        batch (list): Uma lista de tuplas (entrada_numerica, saida_numerica), onde cada
                      entrada/saída_numerica é uma lista de tensores.
        tamanho_vetor_bit (int): A dimensão dos vetores de bit (usada para determinar o tamanho total do passo de tempo).
                      
    Returns:
        tuple: (entradas_padded, saidas_padded, comprimentos_reais)
               entradas_padded: Tensor (batch_size, max_seq_len, tamanho_passo_tempo) para entradas.
               saidas_padded: Tensor (batch_size, max_seq_len, tamanho_passo_tempo) para saídas.
               comprimentos_reais: Lista de comprimentos reais das sequências antes do padding.
    """
    
    entradas_numericas = [item[0] for item in batch]
    saidas_numericas = [item[1] for item in batch]

    comprimentos_entrada = [len(seq) for seq in entradas_numericas]
    comprimentos_saida = [len(seq) for seq in saidas_numericas] # Deveria ser igual ao de entrada para cópia simples

    max_comprimento_entrada = max(comprimentos_entrada)
    max_comprimento_saida = max(comprimentos_saida)

    # O tamanho total da entrada/saída de um passo de tempo para a NTM
    # será o tamanho do vetor de bits + o espaço para os tokens especiais one-hot
    num_tokens_especiais_hot = len(TOKEN_DICT) - 2 # [PAD], [INICIO], [FIM], [COPIA_TASK]
    tamanho_total_passo_tempo = tamanho_vetor_bit + num_tokens_especiais_hot

    # Cria tensores de zeros preenchidos
    entradas_padded = torch.zeros(len(batch), max_comprimento_entrada, tamanho_total_passo_tempo, dtype=torch.float32)
    saidas_padded = torch.zeros(len(batch), max_comprimento_saida, tamanho_total_passo_tempo, dtype=torch.float32)
    
    for i, (entrada_seq, saida_seq) in enumerate(zip(entradas_numericas, saidas_numericas)):
        # Empacota cada vetor numérico na sequência no tensor preenchido
        for j, vetor in enumerate(entrada_seq):
            entradas_padded[i, j, :] = vetor
        for j, vetor in enumerate(saida_seq):
            saidas_padded[i, j, :] = vetor

    return entradas_padded, saidas_padded, comprimentos_entrada


if __name__ == "__main__":
    # Exemplo de como usar o Dataset e DataLoader
    
    # Certifique-se que você já rodou GeradorDataset.py para criar esses arquivos
    caminho_treinamento = 'dataset_treinamento_copia.json'
    caminho_avaliacao = 'dataset_avaliacao_copia.json'
    
    # Definir o tamanho do vetor de bits (deve ser o mesmo usado em GeradorDataset.py)
    TAMANHO_VETOR_BIT = 8 

    print(f"Testando carregamento do dataset de treinamento de: {caminho_treinamento}")
    dataset_treino = DatasetTarefaMaquinaTuring(caminho_treinamento, TAMANHO_VETOR_BIT)
    loader_treino = DataLoader(dataset_treino, batch_size=4, shuffle=True, 
                               collate_fn=lambda b: collate_fn_pad_sequencias(b, TAMANHO_VETOR_BIT))
    
    print(f"Total de exemplos no dataset de treino: {len(dataset_treino)}")
    
    # Testar um batch
    for batch_idx, (entradas, saidas, comprimentos) in enumerate(loader_treino):
        print(f"\nBatch {batch_idx+1}:")
        print(f"  Shape das entradas: {entradas.shape}")
        print(f"  Shape das saídas: {saidas.shape}")
        print(f"  Comprimentos reais das sequências: {comprimentos}")
        
        print("Primeiro vetor de entrada do primeiro exemplo:", entradas[0, 0, :])
        print("Primeiro vetor de saída do primeiro exemplo:", saidas[0, 0, :])
        
        if batch_idx == 0:
            break

    print("\nTeste de carregamento do dataset concluído.")