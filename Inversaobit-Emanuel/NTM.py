import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Memória Externa ------------------
class NTMMemory(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N  # número de locais na memória
        self.M = M  # tamanho de cada vetor
        self.register_buffer("mem_bias", torch.Tensor(N, M).uniform_(-0.1, 0.1))

    def reset(self, batch_size, device):
        return self.mem_bias.clone().unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # [B, N, M]

# ------------------ Cabeça Base ------------------
class NTMHead(nn.Module):
    def __init__(self, controller_size, N, M):
        super().__init__()
        self.N, self.M = N, M
        self.key_layer = nn.Linear(controller_size, M)
        self.beta_layer = nn.Linear(controller_size, 1)

    def addressing(self, memory, key, beta):
        key = key.unsqueeze(1)  # [B, 1, M]
        memory = memory + 1e-8
        key = key + 1e-8
        sim = F.cosine_similarity(memory, key, dim=2)  # [B, N]
        beta = F.softplus(beta).squeeze(-1).unsqueeze(1)  # [B, 1]
        weights = F.softmax(beta * sim, dim=1)  # [B, N]
        return weights

# ------------------ Cabeça de Leitura ------------------
class ReadHead(NTMHead):
    def forward(self, memory, controller_output):
        key = self.key_layer(controller_output)
        beta = self.beta_layer(controller_output)
        w = self.addressing(memory, key, beta)  # [B, N]
        read = torch.bmm(w.unsqueeze(1), memory).squeeze(1)  # [B, M]
        return read, w

# ------------------ Cabeça de Escrita ------------------
class WriteHead(NTMHead):
    def __init__(self, controller_size, N, M):
        super().__init__(controller_size, N, M)
        self.erase_layer = nn.Linear(controller_size, M)
        self.add_layer = nn.Linear(controller_size, M)

    def forward(self, memory, controller_output):
        key = self.key_layer(controller_output)
        beta = self.beta_layer(controller_output)
        w = self.addressing(memory, key, beta)  # [B, N]
        erase = torch.sigmoid(self.erase_layer(controller_output)).unsqueeze(1)  # [B, 1, M]
        add = torch.tanh(self.add_layer(controller_output)).unsqueeze(1)         # [B, 1, M]
        w = w.unsqueeze(-1)  # [B, N, 1]
        memory = memory * (1 - w * erase) + w * add  # [B, N, M]
        return memory, w.squeeze(-1)

# ------------------ Arquitetura NTM ------------------
class NTM(nn.Module):
    def __init__(self, input_size, output_size, controller_size=100, N=128, M=20):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.N = N
        self.M = M

        self.memory = NTMMemory(N, M)
        self.controller = nn.Sequential(
            nn.Linear(input_size + M, controller_size),
            nn.ReLU()
        )
        self.read_head = ReadHead(controller_size, N, M)
        self.write_head = WriteHead(controller_size, N, M)
        self.output_layer = nn.Sequential(
            nn.Linear(M + controller_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, input_seq):
        B, T, _ = input_seq.size()
        device = input_seq.device
        memory = self.memory.reset(B, device)
        read_vector = torch.zeros(B, self.M, device=device)
        outputs = []

        for t in range(T):
            x = input_seq[:, t, :]  # [B, input_size]
            ctrl_input = torch.cat([x, read_vector], dim=1)
            ctrl_out = self.controller(ctrl_input)  # [B, controller_size]

            memory, _ = self.write_head(memory, ctrl_out)
            read_vector, _ = self.read_head(memory, ctrl_out)

            output = self.output_layer(torch.cat([ctrl_out, read_vector], dim=1))  # [B, output_size]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, output_size]
