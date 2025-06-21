import torch
import torch.nn as nn
import torch.nn.functional as F

class NTM(nn.Module):
    def __init__(self, input_dim, output_dim, controller_dim=128, memory_units=128, memory_dim=20, heads=1):
        super(NTM, self).__init__()

        # Embedding para entrada discreta (ex: bits 0/1 ou tokens)
        self.embedding = nn.Embedding(output_dim, input_dim)

        # LSTM Controller: recebe entrada + leitura anterior da memória
        self.controller = nn.LSTMCell(input_dim + memory_dim * heads, controller_dim)

        # Parâmetros da memória
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        self.heads = heads

        # Memória inicial aprendível
        self.initial_memory = nn.Parameter(torch.randn(memory_units, memory_dim))

        # Projeções para cabeças de leitura
        self.key_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.beta_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.gate_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.shift_proj = nn.ModuleList([nn.Linear(controller_dim, 3) for _ in range(heads)])
        self.gamma_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])

        # Projeções para escrita (independentes das de leitura)
        self.write_key_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.write_beta_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.write_gate_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.write_shift_proj = nn.ModuleList([nn.Linear(controller_dim, 3) for _ in range(heads)])
        self.write_gamma_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])

        self.erase_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.add_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])

        # Saída final: concatena estado do LSTM e vetores de leitura
        self.output = nn.Linear(controller_dim + memory_dim * heads, output_dim)

    def init_state(self, batch_size):
        # Cria cópia da memória aprendida para cada elemento do batch
        self.memory = self.initial_memory.unsqueeze(0).repeat(batch_size, 1, 1).detach().clone()
        self.read_weights = torch.zeros(batch_size, self.heads, self.memory_units)
        self.write_weights = torch.zeros(batch_size, self.heads, self.memory_units)
        self.read_vectors = torch.zeros(batch_size, self.heads, self.memory_dim)
        self.controller_state = (
            torch.zeros(batch_size, self.controller.hidden_size),
            torch.zeros(batch_size, self.controller.hidden_size)
        )

    def circular_convolution(self, w, s):
        batch_size, heads, mem_units = w.size()
        shift_range = s.size(-1)
        rolled = torch.stack([torch.roll(w, shifts=j - 1, dims=2) for j in range(shift_range)], dim=-1)
        return torch.sum(rolled * s.unsqueeze(2), dim=-1)

    def address_memory(self, memory, key, beta, g, shift, gamma, prev_weights):
        memory_norm = F.normalize(memory, dim=2)
        key_norm = F.normalize(key, dim=2)
        sim = torch.bmm(memory_norm, key_norm.unsqueeze(-1)).squeeze(-1)
        w_c = F.softmax(beta * sim, dim=-1)
        w_g = g * w_c + (1 - g) * prev_weights
        s = F.softmax(shift, dim=-1)
        w_ = self.circular_convolution(w_g, s)
        w_sharp = w_ ** gamma
        return w_sharp / w_sharp.sum(dim=-1, keepdim=True)

    def write_to_memory(self, weights, erase_vector, add_vector):
        erase = torch.einsum("bhn,bhm->bhnm", weights, erase_vector)
        add = torch.einsum("bhn,bhm->bhnm", weights, add_vector)
        self.memory = self.memory * (1 - erase.sum(dim=1)) + add.sum(dim=1)

    def forward(self, x):
        batch_size, seq_len = x.size()
        self.init_state(batch_size)

        outputs = []
        embedded = self.embedding(x)

        for t in range(seq_len):
            inp = embedded[:, t, :]
            read_flat = self.read_vectors.view(batch_size, -1)
            lstm_input = torch.cat([inp, read_flat], dim=1)
            h, c = self.controller(lstm_input, self.controller_state)
            self.controller_state = (h, c)

            # Cabeças de leitura
            read_keys, read_betas, read_gates, read_shifts, read_gammas = [], [], [], [], []
            write_keys, write_betas, write_gates, write_shifts, write_gammas = [], [], [], [], []
            erase_vecs, add_vecs = [], []

            for i in range(self.heads):
                read_keys.append(self.key_proj[i](h).unsqueeze(1))
                read_betas.append(F.softplus(self.beta_proj[i](h)).unsqueeze(1))
                read_gates.append(torch.sigmoid(self.gate_proj[i](h)).unsqueeze(1))
                read_shifts.append(self.shift_proj[i](h).unsqueeze(1))
                read_gammas.append((1 + F.softplus(self.gamma_proj[i](h))).unsqueeze(1))

                write_keys.append(self.write_key_proj[i](h).unsqueeze(1))
                write_betas.append(F.softplus(self.write_beta_proj[i](h)).unsqueeze(1))
                write_gates.append(torch.sigmoid(self.write_gate_proj[i](h)).unsqueeze(1))
                write_shifts.append(self.write_shift_proj[i](h).unsqueeze(1))
                write_gammas.append((1 + F.softplus(self.write_gamma_proj[i](h))).unsqueeze(1))

                erase_vecs.append(torch.sigmoid(self.erase_proj[i](h)).unsqueeze(1))
                add_vecs.append(torch.tanh(self.add_proj[i](h)).unsqueeze(1))

            # Concatena as heads
            rk, rb, rg, rs, rgm = map(lambda l: torch.cat(l, dim=1), [read_keys, read_betas, read_gates, read_shifts, read_gammas])
            wk, wb, wg, ws, wgm = map(lambda l: torch.cat(l, dim=1), [write_keys, write_betas, write_gates, write_shifts, write_gammas])
            ev = torch.cat(erase_vecs, dim=1)
            av = torch.cat(add_vecs, dim=1)

            self.read_weights = self.address_memory(self.memory, rk, rb, rg, rs, rgm, self.read_weights)
            read_vecs = torch.einsum("bhn,bnm->bhm", self.read_weights, self.memory)
            self.write_weights = self.address_memory(self.memory, wk, wb, wg, ws, wgm, self.write_weights)
            self.write_to_memory(self.write_weights, ev, av)

            self.read_vectors = read_vecs
            read_cat = read_vecs.view(batch_size, -1)
            output_input = torch.cat([h, read_cat], dim=1)
            logits = self.output(output_input)
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)