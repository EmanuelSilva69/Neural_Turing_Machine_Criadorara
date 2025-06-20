import torch
import torch.nn as nn
import torch.nn.functional as F

class NTM(nn.Module):
    def __init__(self, input_dim, output_dim, controller_dim=128, memory_units=128, memory_dim=20, heads=1):
        super(NTM, self).__init__()

        # Embedding (para entrada tokenizada)
        self.embedding = nn.Embedding(output_dim, input_dim)

        # Controller: LSTM
        self.controller = nn.LSTMCell(input_dim + memory_dim * heads, controller_dim)

        # Memória externa
        self.memory = torch.randn(memory_units, memory_dim)
        self.memory_units = memory_units
        self.memory_dim = memory_dim

        # Heads de leitura e escrita
        self.heads = heads
        self.read_weights = torch.zeros(heads, memory_units)
        self.write_weights = torch.zeros(heads, memory_units)
        self.read_vectors = torch.zeros(heads, memory_dim)

        # Parâmetros de endereçamento
        self.key_proj = nn.Linear(controller_dim, memory_dim)
        self.beta_proj = nn.Linear(controller_dim, 1)

        # Parâmetros de escrita
        self.erase_proj = nn.Linear(controller_dim, memory_dim)
        self.add_proj = nn.Linear(controller_dim, memory_dim)

        # Decoder final
        self.output = nn.Linear(controller_dim + memory_dim * heads, output_dim)

    def init_state(self, batch_size):
        # Inicializa estados do LSTM e da memória
        self.memory = torch.randn(self.memory_units, self.memory_dim)
        self.read_weights = torch.zeros(self.heads, self.memory_units)
        self.write_weights = torch.zeros(self.heads, self.memory_units)
        self.read_vectors = torch.zeros(self.heads, self.memory_dim)
        self.controller_state = (
            torch.zeros(batch_size, self.controller.hidden_size),
            torch.zeros(batch_size, self.controller.hidden_size)
        )

    def address_memory(self, key, beta):
        # Similaridade coseno
        memory_norm = F.normalize(self.memory, dim=1)
        key_norm = F.normalize(key, dim=1)
        sim = torch.matmul(memory_norm, key_norm.unsqueeze(-1)).squeeze(-1)
        weights = F.softmax(beta * sim, dim=-1)
        return weights

    def write_to_memory(self, weights, erase_vector, add_vector):
        # erase_vector: [memory_dim], add_vector: [memory_dim], weights: [memory_units]
        erase = torch.ger(weights, erase_vector)  # [N x M]
        add = torch.ger(weights, add_vector)      # [N x M]

        self.memory = self.memory * (1 - erase) + add

    def forward(self, x):
        batch_size, seq_len = x.size()
        self.init_state(batch_size)

        outputs = []
        embedded = self.embedding(x)  # [B, T, input_dim]

        for t in range(seq_len):
            inp = embedded[:, t, :]  # [B, input_dim]
            read_flat = self.read_vectors.view(batch_size, -1)  # [B, heads * mem_dim]
            lstm_input = torch.cat([inp, read_flat], dim=1)

            h, c = self.controller(lstm_input, self.controller_state)
            self.controller_state = (h, c)

            # Endereçamento de leitura
            key = self.key_proj(h)
            beta = F.softplus(self.beta_proj(h))
            self.read_weights = self.address_memory(key, beta)
            self.read_vectors = torch.matmul(self.read_weights, self.memory)

            # Escrita na memória
            self.write_weights = self.read_weights.clone()  # Compartilha os mesmos pesos para simplicidade
            erase_vector = torch.sigmoid(self.erase_proj(h)).squeeze(0)  # [M]
            add_vector = torch.tanh(self.add_proj(h)).squeeze(0)         # [M]
            self.write_to_memory(self.write_weights, erase_vector, add_vector)

            # Combinação e decodifica
            output_input = torch.cat([h, self.read_vectors.view(batch_size, -1)], dim=1)
            logits = self.output(output_input)
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, output_dim]
