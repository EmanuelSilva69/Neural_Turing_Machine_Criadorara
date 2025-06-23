import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedforwardController(nn.Module):
    def __init__(self, input_dim, controller_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, controller_dim),
            nn.ReLU(),
            nn.Linear(controller_dim, controller_dim)
        )

    def forward(self, x, state=None):
        return self.fc(x), None


class NTM(nn.Module):
    def __init__(self, input_dim, output_dim, controller_dim=128, memory_units=128, memory_dim=20, heads=1, controller_type='lstm'):
        super(NTM, self).__init__()

        self.embedding = nn.Embedding(output_dim, input_dim)
        self.controller_type = controller_type
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        self.heads = heads

        self.controller_input_dim = input_dim + memory_dim * heads
        self.controller_dim = controller_dim

        if controller_type == 'lstm':
            self.controller = nn.LSTMCell(self.controller_input_dim, controller_dim)
        elif controller_type == 'feedforward':
            self.controller = FeedforwardController(self.controller_input_dim, controller_dim)
        else:
            raise ValueError("controller_type deve ser 'lstm' ou 'feedforward'")

        self.key_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.beta_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.gate_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.shift_proj = nn.ModuleList([nn.Linear(controller_dim, 3) for _ in range(heads)])
        self.gamma_proj = nn.ModuleList([nn.Linear(controller_dim, 1) for _ in range(heads)])
        self.erase_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])
        self.add_proj = nn.ModuleList([nn.Linear(controller_dim, memory_dim) for _ in range(heads)])

        self.output = nn.Linear(controller_dim + memory_dim * heads, output_dim)

        # ⬇️ Device persistente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_state(self, batch_size):
        self.memory = torch.randn(batch_size, self.memory_units, self.memory_dim, device=self.device) * 0.05
        self.read_weights = torch.zeros(batch_size, self.heads, self.memory_units, device=self.device)
        self.write_weights = torch.zeros(batch_size, self.heads, self.memory_units, device=self.device)
        self.read_vectors = torch.zeros(batch_size, self.heads, self.memory_dim, device=self.device)

        if self.controller_type == 'lstm':
            self.controller_state = (
                torch.zeros(batch_size, self.controller.hidden_size, device=self.device),
                torch.zeros(batch_size, self.controller.hidden_size, device=self.device)
            )
        else:
            self.controller_state = None

    def circular_convolution(self, w, s):
        batch_size, num_heads, mem_units = w.size()
        shift_range = s.size(-1)
        rolled = torch.stack([torch.roll(w, shifts=j - 1, dims=2) for j in range(shift_range)], dim=-1)
        return torch.sum(rolled * s.unsqueeze(2), dim=-1)

    def address_memory(self, key, beta, g, shift, gamma, prev_weights):
        memory_norm = F.normalize(self.memory, dim=2)
        key_norm = F.normalize(key, dim=2).transpose(1, 2)
        sim = torch.bmm(memory_norm, key_norm).transpose(1, 2)
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
        x = x.to(self.device)
        self.init_state(batch_size)
        outputs = []

        embedded = self.embedding(x)  # [B, T, input_dim]

        for t in range(seq_len):
            inp = embedded[:, t, :]
            read_flat = self.read_vectors.view(batch_size, -1)
            controller_input = torch.cat([inp, read_flat], dim=1)

            if self.controller_type == 'lstm':
                h, c = self.controller(controller_input, self.controller_state)
                self.controller_state = (h, c)
            else:
                h, _ = self.controller(controller_input)

            keys = torch.cat([self.key_proj[i](h).unsqueeze(1) for i in range(self.heads)], dim=1)
            betas = torch.cat([F.softplus(self.beta_proj[i](h)).unsqueeze(1) for i in range(self.heads)], dim=1)
            gates = torch.cat([torch.sigmoid(self.gate_proj[i](h)).unsqueeze(1) for i in range(self.heads)], dim=1)
            shifts = torch.cat([self.shift_proj[i](h).unsqueeze(1) for i in range(self.heads)], dim=1)
            gammas = torch.cat([(1 + F.softplus(self.gamma_proj[i](h))).unsqueeze(1) for i in range(self.heads)], dim=1)
            erase_vecs = torch.cat([torch.sigmoid(self.erase_proj[i](h)).unsqueeze(1) for i in range(self.heads)], dim=1)
            add_vecs = torch.cat([torch.tanh(self.add_proj[i](h)).unsqueeze(1) for i in range(self.heads)], dim=1)

            self.read_weights = self.address_memory(keys, betas, gates, shifts, gammas, self.read_weights)
            self.read_vectors = torch.einsum("bhn,bnm->bhm", self.read_weights, self.memory)

            self.write_weights = self.read_weights.clone()
            self.write_to_memory(self.write_weights, erase_vecs, add_vecs)

            read_cat = self.read_vectors.view(batch_size, -1)
            output_input = torch.cat([h, read_cat], dim=1)
            logits = self.output(output_input)
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def collate_fn(batch):
    entradas, saidas = zip(*batch)

    # Encontra o comprimento máximo da batch
    max_input_len = max(len(seq) for seq in entradas)
    max_output_len = max(len(seq) for seq in saidas)

    # Preenche com 0 (ou outro valor, como token de padding)
    padded_inputs = [seq + [0] * (max_input_len - len(seq)) for seq in entradas]
    padded_outputs = [seq + [0] * (max_output_len - len(seq)) for seq in saidas]

    # Converte para tensor
    entradas_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    saidas_tensor = torch.tensor(padded_outputs, dtype=torch.long)

    return entradas_tensor, saidas_tensor