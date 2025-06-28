import json
import random

TOKEN_DICT = {
    '[PAD]': 0,        # Token de preenchimento (padding)
    '[INICIO]': 1,     # Token de início de sequência (opcional, mas comum para tasks)
    '[FIM]': 2,        # Token de fim de sequência (delimitador de entrada)
    '[COPIA_TASK]': 3, # Token para indicar a tarefa de cópia
    '0': 4,            # Bit 0
    '1': 5,            # Bit 1
}

REVERSE_TOKEN_DICT = {v: k for k, v in TOKEN_DICT.items()}
PAD_TOKEN = TOKEN_DICT['[PAD]']
START_TOKEN = TOKEN_DICT['[INICIO]']
END_TOKEN = TOKEN_DICT['[FIM]']
TASK_COPY_TOKEN = TOKEN_DICT['[COPIA_TASK]']


def gerar_exemplo_copia(comprimento_min: int, comprimento_max: int, tamanho_vetor_bit: int = 8):
    """
    Gera um único exemplo de entrada e saída para a tarefa de cópia.
    
    Args:
        comprimento_min (int): Comprimento mínimo da sequência de bits.
        comprimento_max (int): Comprimento máximo da sequência de bits.
        tamanho_vetor_bit (int): Número de bits em cada vetor de entrada (dimensão de cada 'símbolo').
                                 Para a tarefa de cópia básica, geralmente 1 bit por 'símbolo'
                                 mas a NTM lida com vetores.
    
    Returns:
        tuple: (entrada_tokenizada, saida_tokenizada)
               entrada_tokenizada: lista de IDs de token representando a sequência de entrada.
               saida_tokenizada: lista de IDs de token representando a sequência de saída esperada.
    """
    
    # Gera um comprimento aleatório para a sequência dentro do intervalo especificado.
    comprimento_sequencia = random.randint(comprimento_min, comprimento_max)
    
    # Gera a sequência de bits de entrada aleatória.
    # Cada 'símbolo' é um vetor de bits. Para a tarefa de cópia, geralmente é um bit por vez.
    # O paper da NTM usa 8-bit random vectors. Vamos simular isso gerando uma lista de listas.
    
    # Exemplo: [[0,1,0,1,1,0,0,1], [1,1,0,0,1,1,0,1], ...]
    sequencia_entrada_bits = []
    for _ in range(comprimento_sequencia):
        vetor_bit = [random.randint(0, 1) for _ in range(tamanho_vetor_bit)]
        sequencia_entrada_bits.append(vetor_bit)

    # A entrada para a NTM: [COPIA_TASK], sequencia_entrada, [FIM]
    # Representamos os vetores de bits como strings para fácil serialização em JSON,
    # e depois os convertemos de volta para vetores numéricos durante a formatação do dataset.
    entrada_json = [REVERSE_TOKEN_DICT[TASK_COPY_TOKEN]] + \
                   [''.join(map(str, vetor)) for vetor in sequencia_entrada_bits] + \
                   [REVERSE_TOKEN_DICT[END_TOKEN]]
    
    # A saída esperada da NTM: sequencia_entrada, [FIM]
    saida_json = [''.join(map(str, vetor)) for vetor in sequencia_entrada_bits] + \
                 [REVERSE_TOKEN_DICT[END_TOKEN]]
                 
    return entrada_json, saida_json


def gerar_dataset(nome_arquivo: str, num_exemplos: int,
                  comprimento_min_seq: int, comprimento_max_seq: int,
                  tamanho_vetor_bit: int = 8):
    """
    Gera e salva um conjunto de dados de exemplos de cópia em um arquivo JSON.
    
    Args:
        nome_arquivo (str): Nome do arquivo JSON onde o dataset será salvo.
        num_exemplos (int): Número total de exemplos a serem gerados.
        comprimento_min_seq (int): Comprimento mínimo das sequências de bits.
        comprimento_max_seq (int): Comprimento máximo das sequências de bits.
        tamanho_vetor_bit (int): Dimensão de cada vetor de bit na sequência.
    """
    dataset = []
    print(f"Gerando {num_exemplos} exemplos para a tarefa de cópia...")
    for i in range(num_exemplos):
        entrada, saida = gerar_exemplo_copia(comprimento_min_seq, comprimento_max_seq, tamanho_vetor_bit)
        dataset.append({
            'entrada': entrada,
            'saida': saida
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{num_exemplos} exemplos gerados.")

    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset salvo em '{nome_arquivo}' com {len(dataset)} exemplos.")

if __name__ == "__main__":
    # Exemplo de uso para gerar um dataset de treinamento e avaliação
    
    # Dataset de treinamento
    gerar_dataset(
        nome_arquivo='dataset_treinamento_copia.json',
        num_exemplos=10000,
        comprimento_min_seq=1,
        comprimento_max_seq=20, # Comprimentos usados no paper original
        tamanho_vetor_bit=8 # 8-bit random vectors usados no paper
    )

    # Dataset de avaliação (comprimentos possivelmente maiores para testar generalização)
    gerar_dataset(
        nome_arquivo='dataset_avaliacao_copia.json',
        num_exemplos=10000,
        comprimento_min_seq=21, # Maior que o de treinamento para testar generalização
        comprimento_max_seq=50, # Até 50 para teste, ou mais para verificar limites
        tamanho_vetor_bit=8
    )

    # O dicionário de tokens também pode ser salvo para uso futuro
    with open('tokens_copia.json', 'w', encoding='utf-8') as f:
        json.dump(TOKEN_DICT, f, indent=4, ensure_ascii=False)
    print("Dicionário de tokens salvo em 'tokens_copia.json'.")