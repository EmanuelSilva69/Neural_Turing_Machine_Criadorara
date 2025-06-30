import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import math

# Importa as classes e funções que você criou
from NTM import NeuralTuringMachine
from formatacaodataset import DatasetTarefaMaquinaTuring, collate_fn_pad_sequencias, TOKEN_DICT, PAD_TOKEN

# --- Definições de Hiperparâmetros ---
# Estes valores são exemplos e podem precisar de ajuste fino.
# Referências para alguns valores:

NUM_EPOCAS = 10 # Número de épocas de treinamento (pode ser muito alto, ajuste conforme convergência)
TAMANHO_BATCH = 1    # Batch size: Para NTMs com estado de memória que persiste, é comum usar 1.
                     # Se batch_size > 1, a memória da NTM no forward precisaria ser tratada por elemento do batch.
TAXA_APRENDIZADO = 1e-4 # Taxa de aprendizado
CLIP_GRADIENTE = 50  # Valor para o corte de gradientes

# Parâmetros da NTM (devem corresponder ao GeradorDataset e ao seu modelo NTM)
TAMANHO_VETOR_BIT = 8 # Dimensão dos vetores de bit (usado no GeradorDataset)

# O tamanho_entrada da NTM é (tamanho_vetor_bit + número de tokens especiais hot-encoded)
# Tokens especiais: [PAD], [INICIO], [FIM], [COPIA_TASK]
NUM_TOKENS_ESPECIAIS_HOT = len(TOKEN_DICT) - 2 # Excluindo '0' e '1' que são bits
TAMANHO_ENTRADA_NTM = TAMANHO_VETOR_BIT + NUM_TOKENS_ESPECIAIS_HOT
TAMANHO_SAIDA_NTM = TAMANHO_ENTRADA_NTM # A saída da cópia tem a mesma dimensão que a entrada.

TAMANHO_MEMORIA_LINHAS = 128   # N: Número de locais de memória
TAMANHO_MEMORIA_COLUNAS = 20  # M: Dimensão de cada vetor de memória
NUM_CABECAS_LEITURA = 1       # Número de cabeçotes de leitura
NUM_CABECAS_ESCRITA = 1       # Número de cabeçotes de escrita
TAMANHO_CONTROLADOR = 100     # Tamanho do estado oculto do controlador (para LSTM ou Feedforward)
TIPO_CONTROLADOR = 'Feedforward'

CAMINHO_MODELOS_SALVOS = 'modelos_salvos' # Pasta para salvar os modelos
NOME_ARQUIVO_MODELO = os.path.join(CAMINHO_MODELOS_SALVOS, 'ntm_copia_basica.pth')

# --- Configuração do Dispositivo (GPU vs. CPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {DEVICE}")

# --- Carregar Dados ---
print("Carregando datasets...")
dataset_treinamento = DatasetTarefaMaquinaTuring('dataset_treinamento_copia.json', TAMANHO_VETOR_BIT)
loader_treinamento = DataLoader(dataset_treinamento, batch_size=TAMANHO_BATCH, shuffle=True,
                                collate_fn=lambda b: collate_fn_pad_sequencias(b, TAMANHO_VETOR_BIT))

dataset_avaliacao = DatasetTarefaMaquinaTuring('dataset_avaliacao_copia.json', TAMANHO_VETOR_BIT)
loader_avaliacao = DataLoader(dataset_avaliacao, batch_size=TAMANHO_BATCH, shuffle=False, # Não precisa embaralhar na avaliação
                              collate_fn=lambda b: collate_fn_pad_sequencias(b, TAMANHO_VETOR_BIT))

print(f"Dataset de treinamento carregado com {len(dataset_treinamento)} exemplos.")
print(f"Dataset de avaliação carregado com {len(dataset_avaliacao)} exemplos.")

# --- Instanciar Modelo, Otimizador e Função de Perda ---
print("Instanciando modelo, otimizador e função de perda...")
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

# Otimizador Adam é comumente usado em NTMs
otimizador = optim.Adam(modelo_ntm.parameters(), lr=TAXA_APRENDIZADO)

# Função de perda: Binary Cross Entropy Loss.
# A saída da NTM já é passada por sigmoid em NTM.py, então BCELoss é apropriada.
# - Cross-entropy objective function for binary targets.
funcao_perda = nn.BCELoss() 

# Certifique-se de que a pasta para salvar modelos existe
os.makedirs(CAMINHO_MODELOS_SALVOS, exist_ok=True)

# --- Loop de Treinamento ---
print("Iniciando treinamento...")
melhor_perda_avaliacao = float('inf')

for epoca in range(1, NUM_EPOCAS + 1):
    modelo_ntm.train() # Define o modelo para o modo de treinamento
    perda_total_epoca = 0

    for idx_batch, (entradas_batch, saidas_batch, comprimentos_reais) in enumerate(loader_treinamento):
        # inputs_batch: (batch_size, max_seq_len, tamanho_passo_tempo)
        # targets_batch: (batch_size, max_seq_len, tamanho_passo_tempo)
        
        # Redefine os estados internos da NTM para cada nova sequência/exemplo.
        # Isso é crucial para NTMs, pois cada exemplo é um "episódio" independente.
        modelo_ntm.resetar_estados_internos()
        estado_controlador = None # Para LSTM: (h_0, c_0)

        perda_batch = 0
        
        # Itera sobre os passos de tempo da sequência (comprimento máximo após padding)
        for t in range(entradas_batch.size(1)): # size(1) é max_seq_len
            # Pega a entrada e a saída esperada para o passo de tempo atual
            entrada_passo_tempo = entradas_batch[:, t, :].to(DEVICE)
            saida_esperada_passo_tempo = saidas_batch[:, t, :].to(DEVICE)

            # Para a tarefa de cópia, queremos que a NTM aprenda a copiar,
            # então a saída esperada é relevante APENAS após o token de fim ([FIM]).
            # No entanto, a BCELoss vai calcular a perda para todos os passos.
            # Podemos mascarar a perda se quisermos focar apenas na parte de recall.
            # Por simplicidade inicial, calculamos a perda para todos os passos.
            # O paper original calcula a perda por sequência em bits-por-sequência.
            # Uma forma de fazer isso é calcular a perda para cada passo e somar.

            # Forward pass da NTM
            saida_prevista_passo_tempo, estado_controlador = modelo_ntm(entrada_passo_tempo, estado_controlador)
            
            # Ajuste de dimensão para a função de perda, se necessário
            # BCELoss espera (N, *) para inputs e targets.
            # A saida_prevista_passo_tempo é (batch_size, tamanho_saida_ntm)
            # A saida_esperada_passo_tempo é (batch_size, tamanho_saida_ntm)

            # Calcula a perda para o passo de tempo atual
            perda_passo_tempo = funcao_perda(saida_prevista_passo_tempo, saida_esperada_passo_tempo)
            
            # Acumula a perda para o batch
            perda_batch += perda_passo_tempo

        # Backpropagation e otimização para o batch
        otimizador.zero_grad() # Zera os gradientes acumulados
        perda_batch.backward() # Calcula os gradientes
        
        # Corta os gradientes para evitar o problema de gradientes explosivos
        torch.nn.utils.clip_grad_norm_(modelo_ntm.parameters(), CLIP_GRADIENTE)
        
        otimizador.step() # Atualiza os pesos do modelo

        perda_total_epoca += perda_batch.item() # Acumula a perda para a época

    perda_media_epoca = perda_total_epoca / len(loader_treinamento) # Perda média por batch

    print(f"Época {epoca}/{NUM_EPOCAS}, Perda de Treinamento: {perda_media_epoca:.4f}")

    # Avaliação (a cada N épocas ou sempre)
    if epoca % 100 == 0: # Avalia a cada 100 épocas para economizar tempo
        modelo_ntm.eval() # Define o modelo para o modo de avaliação (desliga dropout/batchnorm, etc.)
        perda_avaliacao_total = 0
        with torch.no_grad(): # Desativa o cálculo de gradientes durante a avaliação
            for idx_batch_val, (entradas_val_batch, saidas_val_batch, comprimentos_val_reais) in enumerate(loader_avaliacao):
                modelo_ntm.resetar_estados_internos()
                estado_controlador_val = None
                perda_batch_val = 0
                for t_val in range(entradas_val_batch.size(1)):
                    entrada_val_passo_tempo = entradas_val_batch[:, t_val, :].to(DEVICE)
                    saida_esperada_val_passo_tempo = saidas_val_batch[:, t_val, :].to(DEVICE)

                    saida_prevista_val_passo_tempo, estado_controlador_val = modelo_ntm(entrada_val_passo_tempo, estado_controlador_val)
                    
                    perda_batch_val += funcao_perda(saida_prevista_val_passo_tempo, saida_esperada_val_passo_tempo)
                
                perda_avaliacao_total += perda_batch_val.item()

        perda_media_avaliacao = perda_avaliacao_total / len(loader_avaliacao)
        print(f"  Perda de Avaliação na Época {epoca}: {perda_media_avaliacao:.4f}")

        # Salva o modelo se a perda de avaliação melhorar
        if perda_media_avaliacao < melhor_perda_avaliacao:
            melhor_perda_avaliacao = perda_media_avaliacao
            torch.save(modelo_ntm.state_dict(), NOME_ARQUIVO_MODELO)
            print(f"  Modelo salvo em '{NOME_ARQUIVO_MODELO}' (Perda de Avaliação: {melhor_perda_avaliacao:.4f})")

print("\nTreinamento concluído!")
print(f"Melhor modelo salvo em '{NOME_ARQUIVO_MODELO}'.")