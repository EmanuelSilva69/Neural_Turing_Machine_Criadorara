import torch
import torch.nn as nn
import torch.nn.functional as F

class NTM(nn.Module):
    def __init__(self, input_dim, output_dim, controller_dim=128, memory_units=128, memory_dim=20, heads=1):
        super(NTM, self).__init__()

        # Camada de embedding para transformar tokens em vetores densos
        self.embedding = nn.Embedding(output_dim, input_dim)

        # Controlador LSTM que processa a entrada concatenada com os vetores lidos
        self.controller = nn.LSTMCell(input_dim + memory_dim * heads, controller_dim)

        # Parâmetros da memória
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        self.heads = heads

        # Projeções para endereçamento e escrita, uma por head
        self.key_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.beta_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.gate_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.shift_proj = nn.ModuleList([nn.Linear(controller_dim, 3) for _ in range(heads)])
        self.gamma_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.erase_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.add_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])

        # Camada final para gerar saída a partir do estado do controlador e leitura da memória
        self.output = nn.Linear(controller_dim + memory_dim * heads, output_dim)

    def init_state(self, batch_size):
        # Memória independente para cada elemento do batch
        self.memory = torch.randn(batch_size, self.memory_units, self.memory_dim)

        # Pesos de leitura/escrita e vetores de leitura por exemplo do batch
        self.read_weights = torch.zeros(batch_size, self.heads, self.memory_units)
        self.write_weights = torch.zeros(batch_size, self.heads, self.memory_units)
        self.read_vectors = torch.zeros(batch_size, self.heads, self.memory_dim)

        # Estado do LSTM por exemplo do batch
        self.controller_state = (
            torch.zeros(batch_size, self.controller.hidden_size),
            torch.zeros(batch_size, self.controller.hidden_size)
        )

    def circular_convolution(self, w, s):
        batch_size, num_heads, mem_units = w.size()
        shift_range = s.size(-1)
        rolled = torch.stack([torch.roll(w, shifts=j - 1, dims=2) for j in range(shift_range)], dim=-1)
        return torch.sum(rolled * s.unsqueeze(2), dim=-1)

    def address_memory(self, key, beta, g, shift, gamma, prev_weights):
        # Endereçamento baseado em conteúdo
        memory_norm = F.normalize(self.memory, dim=2)
                # key: [batch, heads, memory_dim]
        key_norm = F.normalize(key, dim=2)  # [B, H, M]
        key_norm = key_norm.transpose(1, 2)  # [B, M, H]
        sim = torch.bmm(memory_norm, key_norm)  # [B, N, H]
        sim = sim.transpose(1, 2)  # [B, H, N]
        w_c = F.softmax(beta * sim, dim=-1)

        # Interpolação
        w_g = g * w_c + (1 - g) * prev_weights

        # Shift circular
        s = F.softmax(shift, dim=-1)
        w_ = self.circular_convolution(w_g, s)

        # Sharpening
        w_sharp = w_ ** gamma
        return w_sharp / w_sharp.sum(dim=-1, keepdim=True)

    def write_to_memory(self, weights, erase_vector, add_vector):
        # Atualização batched da memória com erase e add
        erase = torch.einsum("bhn,bhm->bhnm", weights, erase_vector)
        add = torch.einsum("bhn,bhm->bhnm", weights, add_vector)
        self.memory = self.memory * (1 - erase.sum(dim=1)) + add.sum(dim=1)

    def forward(self, x):
        batch_size, seq_len = x.size()
        self.init_state(batch_size)

        outputs = []
        embedded = self.embedding(x)  # [B, T, input_dim]

        for t in range(seq_len):
            inp = embedded[:, t, :]  # [B, input_dim]
            read_flat = self.read_vectors.view(batch_size, -1)
            lstm_input = torch.cat([inp, read_flat], dim=1)

            h, c = self.controller(lstm_input, self.controller_state)
            self.controller_state = (h, c)

            # Inicializa listas de vetores
            read_vecs = []
            all_keys, all_betas, all_gates, all_shifts, all_gammas = [], [], [], [], []
            all_erase, all_add = [], []

            for i in range(self.heads):
                all_keys.append(self.key_proj[i](h).unsqueeze(1))
                all_betas.append(F.softplus(self.beta_proj[i](h)).unsqueeze(1))
                all_gates.append(torch.sigmoid(self.gate_proj[i](h)).unsqueeze(1))
                all_shifts.append(self.shift_proj[i](h).unsqueeze(1))
                all_gammas.append((1 + F.softplus(self.gamma_proj[i](h))).unsqueeze(1))
                all_erase.append(torch.sigmoid(self.erase_proj[i](h)).unsqueeze(1))
                all_add.append(torch.tanh(self.add_proj[i](h)).unsqueeze(1))

            # Concatena todos os heads
            keys = torch.cat(all_keys, dim=1)
            betas = torch.cat(all_betas, dim=1)
            gates = torch.cat(all_gates, dim=1)
            shifts = torch.cat(all_shifts, dim=1)
            gammas = torch.cat(all_gammas, dim=1)
            erase_vecs = torch.cat(all_erase, dim=1)
            add_vecs = torch.cat(all_add, dim=1)

            # Endereçamento + leitura
            self.read_weights = self.address_memory(keys, betas, gates, shifts, gammas, self.read_weights)
            read_vecs = torch.einsum("bhn,bnm->bhm", self.read_weights, self.memory)

            # Escrita
            self.write_weights = self.read_weights.clone()
            self.write_to_memory(self.write_weights, erase_vecs, add_vecs)

            self.read_vectors = read_vecs

            read_cat = read_vecs.view(batch_size, -1)
            output_input = torch.cat([h, read_cat], dim=1)
            logits = self.output(output_input)
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)
