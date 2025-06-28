import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

# Importa as classes e funções
from NTM import NeuralTuringMachine
from formatacaodataset import DatasetTarefaMaquinaTuring, collate_fn_pad_sequencias, TOKEN_DICT, END_TOKEN, PAD_TOKEN

# Para exibir os tokens de volta em formato legível
REVERSE_TOKEN_DICT = {v: k for k, v in TOKEN_DICT.items()}

# --- Definições de Hiperparâmetros e Configurações (devem corresponder ao train.py) ---
# Estes valores devem ser os mesmos usados durante o treinamento para instanciar o modelo corretamente.
TAMANHO_BATCH_AVALIACAO = 1 # Geralmente 1 para avaliação sequencial de NTMs
TAXA_APRENDIZADO = 1e-4 # Não é usado na avaliação, mas bom para contexto
CLIP_GRADIENTE = 50 # Não é usado na avaliação

# Parâmetros da NTM
TAMANHO_VETOR_BIT = 8 
NUM_TOKENS_ESPECIAIS_HOT = len(TOKEN_DICT) - 2
TAMANHO_ENTRADA_NTM = TAMANHO_VETOR_BIT + NUM_TOKENS_ESPECIAIS_HOT
TAMANHO_SAIDA_NTM = TAMANHO_ENTRADA_NTM 

TAMANHO_MEMORIA_LINHAS = 128
TAMANHO_MEMORIA_COLUNAS = 20
NUM_CABECAS_LEITURA = 1
NUM_CABECAS_ESCRITA = 1
TAMANHO_CONTROLADOR = 100
TIPO_CONTROLADOR = 'Feedforward'

CAMINHO_MODELOS_SALVOS = 'modelos_salvos'
NOME_ARQUIVO_MODELO = os.path.join(CAMINHO_MODELOS_SALVOS, 'ntm_copia_basica.pth') # Certifique-se de que este nome corresponde ao que você salvou

CAMINHO_DATASET_AVALIACAO = 'dataset_avaliacao_copia.json'

# --- Configuração do Dispositivo (GPU vs. CPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {DEVICE}")

# --- Carregar Dados de Avaliação ---
print(f"Carregando dataset de avaliação de: {CAMINHO_DATASET_AVALIACAO}")
dataset_avaliacao = DatasetTarefaMaquinaTuring(CAMINHO_DATASET_AVALIACAO, TAMANHO_VETOR_BIT)
loader_avaliacao = DataLoader(dataset_avaliacao, batch_size=TAMANHO_BATCH_AVALIACAO, shuffle=False,
                              collate_fn=lambda b: collate_fn_pad_sequencias(b, TAMANHO_VETOR_BIT))
print(f"Dataset de avaliação carregado com {len(dataset_avaliacao)} exemplos.")

# --- Instanciar e Carregar Modelo Treinado ---
print("Instanciando e carregando o modelo treinado...")
modelo_ntm = NeuralTuringMachine(
    tamanho_entrada=TAMANHO_ENTRADA_NTM,
    tamanho_saida=TAMANHO_SAIDA_NTM,
    tamanho_memoria_linhas=TAMANHO_MEMORIA_LINHAS,
    tamanho_memoria_colunas=TAMANHO_MEMORIA_COLUNAS,
    num_cabecas_leitura=NUM_CABECAS_LEITURA,
    num_cabecas_escrita=NUM_CABECAS_ESCRITA,
    tamanho_controlador=TAMANHO_CONTROLADOR,
    tipo_controlador=TIPO_CONTROLADOR
).to(DEVICE)

# Carrega os pesos do modelo treinado
if os.path.exists(NOME_ARQUIVO_MODELO):
    modelo_ntm.load_state_dict(torch.load(NOME_ARQUIVO_MODELO, map_location=DEVICE))
    print(f"Modelo carregado com sucesso de: {NOME_ARQUIVO_MODELO}")
else:
    print(f"Erro: Modelo não encontrado em {NOME_ARQUIVO_MODELO}. Certifique-se de ter treinado o modelo primeiro.")
    exit() # Sai se o modelo não for encontrado

modelo_ntm.eval() # Define o modelo para o modo de avaliação

# Função de perda para calcular a perda na avaliação
funcao_perda = nn.BCELoss()

# --- Funções Auxiliares para Avaliação e Visualização ---

def converter_vetor_para_tokens(vetor_tensor: torch.Tensor, tamanho_vetor_bit: int):
    """
    Converte um vetor numérico de saída da NTM de volta para a string de tokens.
    Assume que os bits são as primeiras 'tamanho_vetor_bit' posições e tokens especiais são one-hot depois.
    """
    # Thresholding para converter probabilidades em bits binários (0 ou 1)
    bits_previstos = (vetor_tensor[:tamanho_vetor_bit] > 0.5).int().tolist()
    
    # Verifica se há algum token especial ativado (pode haver mais de um com sigmoid)
    # Pega o token especial com maior probabilidade se houver.
    tokens_especiais_preds = vetor_tensor[tamanho_vetor_bit:].tolist()
    
    # Encontra o índice do token especial mais provável
    if max(tokens_especiais_preds) > 0.5: # Só considera se a probabilidade é alta o suficiente
        idx_token_especial = np.argmax(tokens_especiais_preds)
        
        # Mapeia o índice de volta para o token string
        if idx_token_especial == 0: token_especial_str = '[PAD]'
        elif idx_token_especial == 1: token_especial_str = '[INICIO]'
        elif idx_token_especial == 2: token_especial_str = '[FIM]'
        elif idx_token_especial == 3: token_especial_str = '[COPIA_TASK]'
        else: token_especial_str = '' # Não reconhecido
        
        # Se um token especial dominar, retornamos ele.
        # Para a tarefa de cópia, [FIM] é o mais relevante.
        if token_especial_str == '[FIM]':
            return '[FIM]'
        # Outros tokens especiais geralmente não fazem parte da saída da cópia,
        # exceto se a tarefa for mais complexa.
        # Para simplicidade na cópia, se for um vetor de bits puro e não [FIM], retornamos os bits.
    
    # Se nenhum token especial for dominante ou for um vetor de bits
    return ''.join(map(str, bits_previstos))


def calcular_metricas_sequencia(previsoes_seq: list, targets_seq: list, comprimentos_reais: list):
    """
    Calcula acurácia bit-a-bit e acurácia de sequência.
    
    Args:
        previsoes_seq (list): Lista de tensores de previsões por passo de tempo.
        targets_seq (list): Lista de tensores de targets por passo de tempo.
        comprimentos_reais (list): Comprimentos reais das sequências antes do padding.
        
    Returns:
        tuple: (acuracia_bit_a_bit, acuracia_sequencia_exata)
    """
    total_bits_corretos = 0
    total_bits = 0
    total_sequencias_exatas = 0
    total_sequencias = len(comprimentos_reais)

    for i in range(total_sequencias):
        previsao_item = previsoes_seq[i] # Lista de tensores para 1 sequencia
        target_item = targets_seq[i] # Lista de tensores para 1 sequencia
        comprimento = comprimentos_reais[i]

        # Compara apenas os passos de tempo reais, ignorando o padding
        previsao_real = torch.stack(previsao_item[:comprimento]) # (comprimento, tamanho_passo_tempo)
        target_real = torch.stack(target_item[:comprimento]) # (comprimento, tamanho_passo_tempo)
        
        # Converte probabilidades para binário (0 ou 1) usando threshold de 0.5
        previsao_binaria = (previsao_real > 0.5).float()

        # Compara bit a bit
        bits_corretos_na_seq = (previsao_binaria == target_real).sum().item()
        total_bits_na_seq = target_real.numel() # Número total de elementos (bits) na sequência real

        total_bits_corretos += bits_corretos_na_seq
        total_bits += total_bits_na_seq

        # Verifica se a sequência inteira foi prevista exatamente
        if torch.equal(previsao_binaria, target_real):
            total_sequencias_exatas += 1
            
    acuracia_bit_a_bit = total_bits_corretos / total_bits if total_bits > 0 else 0
    acuracia_sequencia_exata = total_sequencias_exatas / total_sequencias if total_sequencias > 0 else 0
    
    return acuracia_bit_a_bit, acuracia_sequencia_exata


# --- Loop de Avaliação ---
print("\nIniciando avaliação...")
perda_avaliacao_total = 0
todas_previsoes = []
todos_targets = []
todos_comprimentos = []

with torch.no_grad(): # Desativa o cálculo de gradientes para otimização de memória e velocidade
    for idx_batch, (entradas_batch, saidas_batch, comprimentos_reais_batch) in enumerate(loader_avaliacao):
        # inputs_batch: (batch_size, max_seq_len, tamanho_passo_tempo)
        
        # Redefine os estados internos da NTM para cada nova sequência/exemplo.
        # Isso é crucial na avaliação, assim como no treinamento, para manter os episódios independentes.
        modelo_ntm.resetar_estados_internos()
        estado_controlador = None 

        perda_batch_atual = 0
        previsoes_do_batch_seq = [[] for _ in range(TAMANHO_BATCH_AVALIACAO)]
        targets_do_batch_seq = [[] for _ in range(TAMANHO_BATCH_AVALIACAO)]

        for t in range(entradas_batch.size(1)): # Itera sobre os passos de tempo
            entrada_passo_tempo = entradas_batch[:, t, :].to(DEVICE)
            saida_esperada_passo_tempo = saidas_batch[:, t, :].to(DEVICE)

            saida_prevista_passo_tempo, estado_controlador = modelo_ntm(entrada_passo_tempo, estado_controlador)
            
            perda_batch_atual += funcao_perda(saida_prevista_passo_tempo, saida_esperada_passo_tempo).item()

            # Armazena as previsões e targets para cálculo de métricas e visualização
            for i in range(TAMANHO_BATCH_AVALIACAO):
                # Apenas se o passo de tempo estiver dentro do comprimento real da sequência
                if t < comprimentos_reais_batch[i]:
                    previsoes_do_batch_seq[i].append(saida_prevista_passo_tempo[i].cpu())
                    targets_do_batch_seq[i].append(saida_esperada_passo_tempo[i].cpu())

        perda_avaliacao_total += perda_batch_atual
        todas_previsoes.extend(previsoes_do_batch_seq)
        todos_targets.extend(targets_do_batch_seq)
        todos_comprimentos.extend(comprimentos_reais_batch)

# --- Relatório Final da Avaliação ---
perda_media_avaliacao = perda_avaliacao_total / len(loader_avaliacao)
acuracia_bit_a_bit, acuracia_sequencia_exata = calcular_metricas_sequencia(todas_previsoes, todos_targets, todos_comprimentos)

print(f"\n--- Resultados da Avaliação ---")
print(f"Perda Média: {perda_media_avaliacao:.4f}")
print(f"Acurácia Bit-a-Bit: {acuracia_bit_a_bit:.4f}")
print(f"Acurácia de Sequência Exata: {acuracia_sequencia_exata:.4f}") # Quão frequentemente a sequência INTEIRA foi prevista corretamente

# --- Exemplos de Saída ---
print("\n--- Exemplos de Previsões ---")
num_exemplos_para_mostrar = 5
for i in range(min(num_exemplos_para_mostrar, len(dataset_avaliacao))):
    print(f"\nExemplo {i+1}:")
    
    # Pega os dados originais do dataset
    entrada_original, saida_original = dataset_avaliacao[i]
    comprimento_real = len(entrada_original) # Para a cópia, entrada e saída têm o mesmo comprimento real

    # Pega as previsões correspondentes (que já estão sem padding até o comprimento real)
    previsoes_exemplo = todas_previsoes[i]
    targets_exemplo = todos_targets[i]

    # Converte os tensores de volta para formato de string legível
    entrada_str = [converter_vetor_para_tokens(v, TAMANHO_VETOR_BIT) for v in entrada_original]
    saida_esperada_str = [converter_vetor_para_tokens(v, TAMANHO_VETOR_BIT) for v in targets_exemplo]
    saida_prevista_str = [converter_vetor_para_tokens(v, TAMANHO_VETOR_BIT) for v in previsoes_exemplo]

    # Ajusta a representação do token [FIM] para melhorar a leitura
    # O [FIM] na entrada é um delimitador de input. Na saída, é o fim da cópia.
    
    print(f"  Entrada (Tokens): {' '.join(entrada_str)}")
    print(f"  Saída Esperada (Tokens): {' '.join(saida_esperada_str)}")
    print(f"  Saída Prevista (Tokens): {' '.join(saida_prevista_str)}")

print("\nAvaliação concluída!")