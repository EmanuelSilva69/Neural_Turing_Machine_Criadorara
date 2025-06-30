import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Similaridade cosseno
def cosine_similarity(key, memory):
    key = key / (key.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    memory = memory / (memory.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(key, memory.transpose(-1, -2))

# -- Cabeçote de leitura/escrita
class NTMHead(nn.Module):
    def __init__(self, memory_size, word_size):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size

    def addressing(self, key, beta, memory):
        sim = cosine_similarity(key, memory)
        return F.softmax(beta * sim, dim=-1)

# -- Memória
class NTMMemory(nn.Module):
    def __init__(self, memory_size, word_size):
        super().__init__()
        self.N = memory_size
        self.M = word_size
        self.register_buffer('memory', torch.full((self.N, self.M), 1e-6))

    def reset(self):
        self.memory = torch.full((self.N, self.M), 1e-6, device=self.memory.device)

    def read(self, w):
        return torch.matmul(w.to(self.memory.device), self.memory)

    def write(self, w, e, a):
        with torch.no_grad():
            erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(0))
            add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(0))
            self.memory = self.memory * (1 - erase) + add

# -- Controlador LSTM
class NTMControllerLSTM(nn.Module):
    def __init__(self, input_size, controller_size, output_size, param_size):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, controller_size)
        self.fc_out = nn.Linear(controller_size, output_size)
        self.param_gen = nn.Linear(controller_size, param_size)
        self.controller_size = controller_size

    def forward(self, x, state):
        hx, cx = self.lstm(x, state)
        out = self.fc_out(hx)
        params = self.param_gen(hx)
        return out, params, (hx, cx)

# -- Controlador Feedforward
class NTMControllerFF(nn.Module):
    def __init__(self, input_size, controller_size, output_size, param_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, controller_size),
            nn.ReLU(),
            nn.Linear(controller_size, controller_size),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(controller_size, output_size)
        self.param_gen = nn.Linear(controller_size, param_size)

    def forward(self, x, _):
        h = self.net(x)
        out = self.fc_out(h)
        params = self.param_gen(h)
        return out, params, None

# -- Máquina de Turing Neural
class NeuralTuringMachine(nn.Module):
    def __init__(self, input_size, output_size, controller_size, memory_size, word_size,
                 num_read_heads=1, num_write_heads=1, controller_type='lstm'):
        super().__init__()

        self.word_size = word_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.memory = NTMMemory(memory_size, word_size)

        ctrl_input_size = input_size + num_read_heads * word_size
        total_params = num_write_heads * (word_size + 1 + 2 * word_size) + num_read_heads * (word_size + 1)

        if controller_type == 'lstm':
            self.controller = NTMControllerLSTM(ctrl_input_size, controller_size, output_size, total_params)
            self.is_lstm = True
        else:
            self.controller = NTMControllerFF(ctrl_input_size, controller_size, output_size, total_params)
            self.is_lstm = False

        self.read_heads = nn.ModuleList([NTMHead(memory_size, word_size) for _ in range(num_read_heads)])
        self.write_heads = nn.ModuleList([NTMHead(memory_size, word_size) for _ in range(num_write_heads)])

        self.register_buffer('read_vectors', torch.zeros(num_read_heads, word_size))
        self.controller_state = None

    def reset(self, batch_size, device):
        self.memory.reset()
        self.read_vectors.copy_(torch.zeros_like(self.read_vectors))
        if self.is_lstm:
            h_size = self.controller.controller_size
            self.controller_state = (
                torch.zeros(batch_size, h_size, device=device),
                torch.zeros(batch_size, h_size, device=device)
            )
        else:
            self.controller_state = None

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        read_cat = self.read_vectors.view(1, -1).expand(batch_size, -1).to(device)
        controller_input = torch.cat([x, read_cat], dim=-1)
        ctrl_out, head_params, self.controller_state = self.controller(controller_input, self.controller_state)

        wp_size = self.word_size + 1 + 2 * self.word_size
        rp_size = self.word_size + 1

        wp = self.num_write_heads * wp_size
        write_params = head_params[:, :wp].view(batch_size, self.num_write_heads, wp_size)
        read_params  = head_params[:, wp:].view(batch_size, self.num_read_heads, rp_size)

        # Escrita
        for b in range(batch_size):
            for i in range(self.num_write_heads):
                p = write_params[b, i]
                k = torch.tanh(p[0:self.word_size])
                beta = F.softplus(p[self.word_size:self.word_size+1])
                e = torch.sigmoid(p[self.word_size+1:self.word_size*2+1])
                a = torch.tanh(p[self.word_size*2+1:])
                w = self.write_heads[i].addressing(k.unsqueeze(0), beta.unsqueeze(0), self.memory.memory)
                self.memory.write(w[0], e, a)

        # Leitura
        new_reads = []
        for b in range(batch_size):
            for i in range(self.num_read_heads):
                p = read_params[b, i]
                k = torch.tanh(p[0:self.word_size])
                beta = F.softplus(p[self.word_size:self.word_size+1])
                w = self.read_heads[i].addressing(k.unsqueeze(0), beta.unsqueeze(0), self.memory.memory)
                r = self.memory.read(w[0])
                new_reads.append(r)

        stacked = torch.stack(new_reads[:self.num_read_heads], dim=0)
        with torch.no_grad():
            self.read_vectors.copy_(stacked.detach())

        return ctrl_out, stacked  # retorno explícito da leitura
