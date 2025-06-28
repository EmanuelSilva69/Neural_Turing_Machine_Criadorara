import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralTuringMachine(nn.Module):
    def __init__(self,
                 tamanho_entrada: int,      # Dimensão do vetor de entrada para a NTM
                 tamanho_saida: int,        # Dimensão do vetor de saída da NTM
                 tamanho_memoria_linhas: int, # N - número de locais de memória (linhas)
                 tamanho_memoria_colunas: int, # M - dimensão de cada vetor de memória (colunas)
                 num_cabecas_leitura: int,  # Número de cabeçotes de leitura
                 num_cabecas_escrita: int,  # Número de cabeçotes de escrita
                 tamanho_controlador: int,  # Dimensão interna do controlador (estado oculto)
                 tipo_controlador: str = 'Feedforward'): # Tipo de controlador ('LSTM' ou 'Feedforward')
        super(NeuralTuringMachine, self).__init__()

        self.tamanho_entrada = tamanho_entrada
        self.tamanho_saida = tamanho_saida
        self.tamanho_memoria_linhas = tamanho_memoria_linhas
        self.tamanho_memoria_colunas = tamanho_memoria_colunas
        self.num_cabecas_leitura = num_cabecas_leitura
        self.num_cabecas_escrita = num_cabecas_escrita
        self.tamanho_controlador = tamanho_controlador
        self.tipo_controlador = tipo_controlador

        # Parâmetros de deslocamento (shift) para a convolução.
        self.intervalo_shift = 1 # Define o intervalo de shift (ex: 1 para [-1, 0, 1])
        self.tamanho_kernel_shift = (self.intervalo_shift * 2) + 1 # Tamanho do kernel de convolução (ex: 3)

        # 1. Definir o Controlador
        # A entrada do controlador será a entrada externa concatenada com os vetores lidos
        # na etapa de tempo anterior (r_t-1 de cada cabeçote de leitura).
        tamanho_entrada_controlador = tamanho_entrada + (num_cabecas_leitura * tamanho_memoria_colunas)

        if tipo_controlador == 'LSTM':
            self.controlador = nn.LSTMCell(tamanho_entrada_controlador, tamanho_controlador)
        elif tipo_controlador == 'Feedforward':
            # Implementação do controlador Feedforward
            self.controlador = nn.Linear(tamanho_entrada_controlador, tamanho_controlador)
        else:
            raise ValueError("Tipo de controlador inválido. Escolha 'LSTM' ou 'Feedforward'.")

        # 2. Definir a Matriz de Memória
        # A memória é um parâmetro aprendível. É crucial inicializá-la com pequenos valores constantes
        # para estabilidade durante o treinamento.
        self.memoria = nn.Parameter(torch.Tensor(tamanho_memoria_linhas, tamanho_memoria_colunas))
        nn.init.constant_(self.memoria, 1e-6) # Inicialização para pequenos valores constantes.

        # 3. Definir as Projeções dos Parâmetros dos Cabeçotes a partir da Saída do Controlador
        # O controlador produzirá um vetor grande que será dividido para parametrizar todos os cabeçotes.
        # Para CADA cabeçote (leitura ou escrita), precisamos dos seguintes parâmetros:
        # - k (chave de busca de conteúdo): tamanho_memoria_colunas
        # - beta (força da chave): 1 (escalar)
        # - g (gate de interpolação): 1 (escalar, entre 0 e 1)
        # - s (vetor de shift): tamanho_kernel_shift (distribuição sobre shifts)
        # - gamma (nitidez do foco): 1 (escalar, >= 1)
        
        tamanho_parametros_comuns_por_cabeca = (self.tamanho_memoria_colunas + # k
                                                1 + # beta
                                                1 + # g
                                                self.tamanho_kernel_shift + # s
                                                1)   # gamma

        # Para CADA CABEÇOTE DE ESCRITA, adicionamos parâmetros específicos:
        # - e (vetor de apagamento): tamanho_memoria_colunas (elementos entre 0 e 1)
        # - a (vetor de adição): tamanho_memoria_colunas (elementos entre -1 e 1)
        tamanho_parametros_escrita_adicionais = (self.tamanho_memoria_colunas + # e
                                                 self.tamanho_memoria_colunas) # a

        # Tamanho total da saída de projeção do controlador para todos os cabeçotes
        tamanho_saida_parametros_total = (num_cabecas_leitura * tamanho_parametros_comuns_por_cabeca +
                                          num_cabecas_escrita * (tamanho_parametros_comuns_por_cabeca + tamanho_parametros_escrita_adicionais))

        self.projecao_parametros_cabecotes = nn.Linear(tamanho_controlador, tamanho_saida_parametros_total)
        
        # Projeção final da saída do controlador para a saída externa da NTM
        self.projecao_saida_final = nn.Linear(tamanho_controlador, tamanho_saida)

        # Inicializa os estados internos (para LSTM) e vetores de leitura/pesos iniciais
        self.resetar_estados_internos()

    def resetar_estados_internos(self):
        # Resetar os estados ocultos (h) e de célula (c) do controlador (se for LSTM)
        self.estado_oculto_controlador = None
        self.estado_celula_controlador = None

        # Inicializa os vetores de leitura anteriores (r_0) e os pesos de atenção iniciais (w_0).
        # O paper "Implementing Neural Turing Machines" sugere aprender
        # estes como vetores de bias, mas para o reset, os inicializamos como zeros ou uniformes.
        
        # Vetor de leitura anterior para cada cabeçote, inicialmente zeros.
        # (batch_size, tamanho_memoria_colunas) - assume batch_size=1 para o reset.
        self.vetores_leitura_anteriores = [torch.zeros(1, self.tamanho_memoria_colunas)
                                            for _ in range(self.num_cabecas_leitura)]
        
        # Pesos de atenção anteriores para cada cabeçote (leitura e escrita), inicialmente uniformes.
        # (batch_size, tamanho_memoria_linhas) - assume batch_size=1 para o reset.
        self.pesos_anteriores_leitura = [F.softmax(torch.zeros(1, self.tamanho_memoria_linhas), dim=1)
                                          for _ in range(self.num_cabecas_leitura)]
        self.pesos_anteriores_escrita = [F.softmax(torch.zeros(1, self.tamanho_memoria_linhas), dim=1)
                                          for _ in range(self.num_cabecas_escrita)]


    def forward(self, entrada_externa, estado_controlador_anterior=None):
        # entrada_externa: tensor de entrada externa para a NTM (batch_size, tamanho_entrada)
        # estado_controlador_anterior: tupla (h_anterior, c_anterior) para o controlador LSTM (batch_size, tamanho_controlador)

        batch_size = entrada_externa.size(0)

        # 1. Preparar a entrada para o controlador
        # Concatena a entrada externa com os vetores lidos na etapa de tempo anterior.
        vetores_lidos_concatenados = torch.cat(self.vetores_leitura_anteriores, dim=1)
        entrada_controlador = torch.cat([entrada_externa, vetores_lidos_concatenados], dim=1)

        # 2. Executar o Controlador
        if self.tipo_controlador == 'LSTM':
        # ... Lógica LSTM existente ...
            if estado_controlador_anterior is None:
                h_anterior = torch.zeros(batch_size, self.tamanho_controlador, device=entrada_externa.device)
                c_anterior = torch.zeros(batch_size, self.tamanho_controlador, device=entrada_externa.device)
            else:
                h_anterior, c_anterior = estado_controlador_anterior

            self.estado_oculto_controlador, self.estado_celula_controlador = self.controlador(
                entrada_controlador, (h_anterior, c_anterior)
            )
            saida_controlador = self.estado_oculto_controlador
            estado_controlador_para_proximo = (self.estado_oculto_controlador, self.estado_celula_controlador)
        else: # Feedforward
            saida_controlador = self.controlador(entrada_controlador)
            estado_controlador_para_proximo = None # Não há estado para passar para frente em um FF

        # 3. Gerar os Parâmetros dos Cabeçotes a partir da Saída do Controlador
        # Projeta a saída do controlador para um vetor grande contendo todos os parâmetros dos cabeçotes.
        parametros_brutos_cabecotes = self.projecao_parametros_cabecotes(saida_controlador)
        
        # Divide o vetor de parâmetros brutos em parâmetros individuais para cada cabeçote.
        offset = 0
        parametros_cabecas = [] # Lista de dicionários, um para cada cabeçote

        # Processa parâmetros comuns para todos os cabeçotes (leitura e escrita)
        for _ in range(self.num_cabecas_leitura + self.num_cabecas_escrita):
            # k (chave): tanh é aplicado para mapear para o intervalo [-1, 1]
            k = torch.tanh(parametros_brutos_cabecotes[:, offset : offset + self.tamanho_memoria_colunas])
            offset += self.tamanho_memoria_colunas

            # beta (força da chave): softplus para garantir beta >= 0
            beta = F.softplus(parametros_brutos_cabecotes[:, offset : offset + 1])
            offset += 1

            # g (gate de interpolação): sigmoid para garantir g entre 0 e 1
            g = torch.sigmoid(parametros_brutos_cabecotes[:, offset : offset + 1])
            offset += 1

            # s (vetor de shift): softmax para garantir uma distribuição de probabilidade
            s = F.softmax(parametros_brutos_cabecotes[:, offset : offset + self.tamanho_kernel_shift], dim=1)
            offset += self.tamanho_kernel_shift

            # gamma (nitidez): softplus + 1 para garantir gamma >= 1
            gamma = F.softplus(parametros_brutos_cabecotes[:, offset : offset + 1]) + 1
            offset += 1
            
            parametros_cabecas.append({'k': k, 'beta': beta, 'g': g, 's': s, 'gamma': gamma})

        # Processa parâmetros adicionais específicos para cabeçotes de escrita
        for i in range(self.num_cabecas_leitura, self.num_cabecas_leitura + self.num_cabecas_escrita):
            # e (vetor de apagamento): sigmoid para garantir elementos entre 0 e 1
            e = torch.sigmoid(parametros_brutos_cabecotes[:, offset : offset + self.tamanho_memoria_colunas])
            offset += self.tamanho_memoria_colunas

            # a (vetor de adição): tanh para garantir elementos entre -1 e 1
            a = torch.tanh(parametros_brutos_cabecotes[:, offset : offset + self.tamanho_memoria_colunas])
            offset += self.tamanho_memoria_colunas
            
            parametros_cabecas[i]['e'] = e
            parametros_cabecas[i]['a'] = a

        # 4. Operações de Leitura
        vetores_lidos_atuais = []
        for i in range(self.num_cabecas_leitura):
            params_leitura = parametros_cabecas[i] # Pega os parâmetros do cabeçote de leitura `i`
            
            # Calcula os pesos de atenção (w_t) para o cabeçote de leitura
            pesos_atuais_leitura = self._calcular_pesos_atencao(
                params_leitura['k'], params_leitura['beta'], params_leitura['g'],
                params_leitura['s'], params_leitura['gamma'],
                self.memoria, self.pesos_anteriores_leitura[i]
            )
            
            # Realiza a operação de leitura na memória
            vetor_lido = self._ler_memoria(self.memoria, pesos_atuais_leitura)
            vetores_lidos_atuais.append(vetor_lido)
            self.pesos_anteriores_leitura[i] = pesos_atuais_leitura # Atualiza os pesos para a próxima etapa

        self.vetores_leitura_anteriores = vetores_lidos_atuais # Armazena para a próxima iteração do forward

        # 5. Operações de Escrita
        # Clona a memória atual para realizar as operações de escrita, mantendo o grafo computacional.
        memoria_atualizada = self.memoria.clone()
        for i in range(self.num_cabecas_escrita):
            # Pega os parâmetros do cabeçote de escrita. O índice é ajustado porque `parametros_cabecas`
            # contém primeiro os cabeçotes de leitura e depois os de escrita.
            params_escrita = parametros_cabecas[self.num_cabecas_leitura + i]
            
            # Calcula os pesos de atenção (w_t) para o cabeçote de escrita
            pesos_atuais_escrita = self._calcular_pesos_atencao(
                params_escrita['k'], params_escrita['beta'], params_escrita['g'],
                params_escrita['s'], params_escrita['gamma'],
                memoria_atualizada, self.pesos_anteriores_escrita[i]
            )
            
            # Realiza a operação de escrita na memória (erase e add)
            memoria_atualizada = self._escrever_memoria(
                memoria_atualizada, pesos_atuais_escrita, params_escrita['e'], params_escrita['a'], batch_size
            )
            self.pesos_anteriores_escrita[i] = pesos_atuais_escrita # Atualiza os pesos para a próxima etapa

        self.memoria = nn.Parameter(memoria_atualizada) # Atualiza a memória principal do modelo com a nova versão

        # 6. Saída Final da NTM
        # A saída externa da NTM é uma projeção da saída do controlador, passada por uma sigmoid para bits.
        saida_final = torch.sigmoid(self.projecao_saida_final(saida_controlador))

        # Retorna a saída externa e o estado do controlador (ajustado para FF)
        return saida_final, estado_controlador_para_proximo

    # Métodos Auxiliares para as Operações de Memória
    
    def _similaridade_cosseno(self, u, v_memoria):
        # u: chave (batch_size, tamanho_memoria_colunas)
        # v_memoria: a matriz de memória (tamanho_memoria_linhas, tamanho_memoria_colunas)
        # Retorna: similaridade (batch_size, tamanho_memoria_linhas)
        
        # Normaliza os vetores de chave e os vetores de memória.
        u_norm = F.normalize(u, p=2, dim=1) # Normaliza chaves (batch_size, M)
        v_memoria_norm = F.normalize(v_memoria, p=2, dim=1) # Normaliza linhas da memória (N, M)

        # Calcula o produto escalar entre cada chave e cada linha da memória.
        # (batch_size, M) @ (M, N) = (batch_size, N)
        return torch.matmul(u_norm, v_memoria_norm.transpose(0, 1))

    def _calcular_pesos_atencao(self, k, beta, g, s, gamma, memoria, pesos_anteriores):
        # k: (batch_size, tamanho_memoria_colunas) - chave de busca
        # beta: (batch_size, 1) - força da chave
        # g: (batch_size, 1) - gate de interpolação
        # s: (batch_size, tamanho_kernel_shift) - vetor de deslocamento
        # gamma: (batch_size, 1) - nitidez
        # memoria: (tamanho_memoria_linhas, tamanho_memoria_colunas) - a matriz de memória
        # pesos_anteriores: (batch_size, tamanho_memoria_linhas) - pesos da etapa de tempo anterior

        batch_size = k.size(0)

        # 1. Focagem Baseada em Conteúdo (w_c)
        # Calcula a similaridade de cosseno entre a chave 'k' e cada linha da memória.
        similaridade = self._similaridade_cosseno(k, memoria)
        
        # Aplica beta para ajustar a nitidez e então softmax para normalizar, resultando em w_c.
        w_c = F.softmax(beta * similaridade, dim=1)

        # 2. Interpolação (w_g)
        # Mistura os pesos baseados em conteúdo (`w_c`) com os pesos da etapa anterior (`pesos_anteriores`)
        # usando o gate de interpolação `g`.
        w_g = g * w_c + (1 - g) * pesos_anteriores

        # 3. Deslocamento Convolucional (w_tilde)
        # Realiza uma convolução circular para "deslocar" os pesos `w_g` de acordo com `s`.
        
        # Expande o kernel de shift `s` para o formato (batch_size, 1, tamanho_kernel_shift) para F.conv1d.
        s_kernel = s.unsqueeze(1)

        # Expande `w_g` para o formato (batch_size, 1, tamanho_memoria_linhas) para F.conv1d.
        w_g_expanded = w_g.unsqueeze(1)

        # Para realizar a convolução circular, precisamos adicionar padding circular manualmente.
        # Concatenamos as partes finais do tensor no início e as partes iniciais no final.
        w_g_padded = torch.cat([w_g_expanded[:, :, -self.intervalo_shift:],
                                  w_g_expanded,
                                  w_g_expanded[:, :, :self.intervalo_shift]], dim=2)
        
        # Aplica a convolução 1D. `groups=batch_size` aplica um kernel diferente para cada item do batch.
        w_tilde = F.conv1d(w_g_padded, s_kernel, groups=batch_size).squeeze(1) # Remove a dimensão do canal (1)

        # 4. Afiamento/Nitidez (Sharpening) (w_final)
        # Aplica o fator `gamma` para tornar os pesos mais nítidos e re-normaliza com softmax.
        w_final_numerador = w_tilde.pow(gamma)
        w_final_denominador = w_final_numerador.sum(dim=1, keepdim=True)
        # Adiciona um pequeno epsilon (1e-8) para evitar divisão por zero.
        w_final = w_final_numerador / (w_final_denominador + 1e-8)

        return w_final

    def _ler_memoria(self, memoria, pesos_atencao):
        # memoria: a matriz de memória (tamanho_memoria_linhas, tamanho_memoria_colunas)
        # pesos_atencao: pesos de atenção para a leitura (batch_size, tamanho_memoria_linhas)
        # Retorna: o vetor lido (r_t) (batch_size, tamanho_memoria_colunas)

        # O vetor lido `r_t` é uma combinação convexa das linhas da memória, ponderada pelos pesos de atenção.
        # r_t = sum_i w_t(i) * M_t(i)
        
        # Expande `pesos_atencao` para (batch_size, N, 1) para permitir broadcasting com a memória.
        pesos_expandidos = pesos_atencao.unsqueeze(2)
        
        # Expande `memoria` para (1, N, M) para permitir broadcasting com o batch.
        memoria_expandida = memoria.unsqueeze(0)
        
        # Multiplicação elemento a elemento e soma ao longo da dimensão dos locais de memória (N).
        vetor_lido = (pesos_expandidos * memoria_expandida).sum(dim=1)
        
        return vetor_lido

    def _escrever_memoria(self, memoria_anterior, pesos_atencao, vetor_apagar, vetor_adicionar, batch_size):
        # memoria_anterior: matriz de memória na etapa anterior (tamanho_memoria_linhas, tamanho_memoria_colunas)
        # pesos_atencao: pesos de atenção para a escrita (batch_size, tamanho_memoria_linhas)
        # vetor_apagar: vetor de apagamento (batch_size, tamanho_memoria_colunas) - e_t
        # vetor_adicionar: vetor de adição (batch_size, tamanho_memoria_colunas) - a_t
        # Retorna: a memória atualizada (M_t) (tamanho_memoria_linhas, tamanho_memoria_colunas)

        # Operação de apagamento: M_t_tilde(i) = M_t-1(i) [1 - w_t(i) * e_t]
        # Operação de adição: M_t(i) = M_t_tilde(i) + w_t(i) * a_t
        
        # Expande os tensores para permitir broadcasting para operações em batch.
        pesos_expandidos = pesos_atencao.unsqueeze(2) # (batch_size, N, 1)
        vetor_apagar_expandido = vetor_apagar.unsqueeze(1) # (batch_size, 1, M)
        vetor_adicionar_expandido = vetor_adicionar.unsqueeze(1) # (batch_size, 1, M)
        memoria_expandida = memoria_anterior.unsqueeze(0) # (1, N, M) para broadcast com batch

        # Calcular a máscara de apagamento (w_t(i) * e_t(j) para cada elemento (i,j))
        mascara_apagamento = pesos_expandidos * vetor_apagar_expandido # (batch_size, N, M)
        
        # Aplicar a operação de apagamento (multiplicação elemento a elemento)
        memoria_apos_apagamento = memoria_expandida * (1 - mascara_apagamento) # (batch_size, N, M)

        # Calcular a máscara de adição (w_t(i) * a_t(j) para cada elemento (i,j))
        mascara_adicao = pesos_expandidos * vetor_adicionar_expandido # (batch_size, N, M)
        
        # Aplicar a operação de adição
        memoria_atualizada_batch = memoria_apos_apagamento + mascara_adicao # (batch_size, N, M)
        
        # Como self.memoria é uma única matriz (N, M), precisamos colapsar o batch.
        # Em cenários de treinamento de NTM, é comum usar batch_size=1 para manter o estado sequencial da memória.
        # Se batch_size > 1, essa função precisa retornar a média da memória atualizada no batch,
        # ou o loop de treinamento deve iterar com batch_size=1.
        # Para este caso, assumimos que se o batch_size for 1, retornamos o tensor sem a dimensão do batch.
        # Caso contrário (batch_size > 1), retornamos o tensor (batch_size, N, M) e o chamador decide como agregá-lo
        # de volta a uma única memória do modelo se necessário (o que geralmente não é o caso para NTMs).
        
        return memoria_atualizada_batch.squeeze(0) if batch_size == 1 else memoria_atualizada_batch