# ntm_inverter_full.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Controller LSTM
class Controller(nn.Module):
    def __init__(self, input_size, output_size, controller_size, memory_size):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size + memory_size, controller_size)
        self.fc = nn.Linear(controller_size, output_size)

    def forward(self, x, prev_state):
        h, c = self.lstm(x, prev_state)
        out = self.fc(h)
        return out, (h, c)

# Neural Turing Machine
class NTM(nn.Module):
    def __init__(self, input_size, output_size, controller_size, N=128, M=20):
        super().__init__()
        self.N = N
        self.M = M
        self.controller = Controller(input_size, output_size, controller_size, memory_size=M)
        self.register_buffer('memory', torch.zeros(N, M))
        self.read_vector = torch.zeros(M)
        self.read_weights = F.softmax(torch.zeros(N), dim=0)
        self.erase = nn.Linear(controller_size, M)
        self.add = nn.Linear(controller_size, M)
        self.key = nn.Linear(controller_size, M)
        self.beta = nn.Linear(controller_size, 1)

    def cosine_similarity(self, key):
        k = key / (key.norm() + 1e-8)
        mem_norm = self.memory / (self.memory.norm(dim=1, keepdim=True) + 1e-8)
        return torch.matmul(mem_norm, k)

    def address_memory(self, h):
        key = torch.tanh(self.key(h))
        beta = F.softplus(self.beta(h)) + 1e-8
        sim = self.cosine_similarity(key)
        w = F.softmax(beta * sim, dim=0)
        return w

    def read(self, w):
        return torch.matmul(w.unsqueeze(0), self.memory).squeeze(0)

    def write(self, w, h):
        erase = torch.sigmoid(self.erase(h))
        add = torch.tanh(self.add(h))
        self.memory = self.memory * (1 - w.unsqueeze(1) * erase.unsqueeze(0)) + w.unsqueeze(1) * add.unsqueeze(0)

    def forward(self, x, prev_state):
        x = torch.cat([x, self.read_vector], dim=-1)
        controller_out, state = self.controller(x, prev_state)
        h = state[0]
        w = self.address_memory(h)
        self.write(w, h)
        self.read_weights = w
        self.read_vector = self.read(w)
        return torch.sigmoid(controller_out), state

# Hiperparâmetros
input_size = 8
output_size = 8
controller_size = 100
seq_len = 10
epochs = 3000

ntm = NTM(input_size, output_size, controller_size)
optimizer = optim.Adam(ntm.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# Treinamento
for epoch in range(epochs):
    input_seq = torch.randint(0, 2, (seq_len, input_size)).float()
    target_seq = 1.0 - input_seq.clone()  # Inversão dos bits

    state = (
        torch.zeros(controller_size),
        torch.zeros(controller_size)
    )

    ntm.memory = torch.zeros(ntm.N, ntm.M)
    ntm.read_vector = torch.zeros(ntm.M)

    outputs = []
    for t in range(seq_len):
        _, state = ntm(input_seq[t], state)

    blank = torch.zeros(input_size)
    for t in range(seq_len):
        out, state = ntm(blank, state)
        outputs.append(out)

    outputs = torch.stack(outputs)
    loss = loss_fn(outputs, target_seq)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Teste final
print("\nTeste final:")
with torch.no_grad():
    test_seq = torch.randint(0, 2, (seq_len, input_size)).float()
    target_seq = 1.0 - test_seq.clone()

    print("Entrada:")
    print(test_seq.int())

    state = (
        torch.zeros(controller_size),
        torch.zeros(controller_size)
    )
    ntm.memory = torch.zeros(ntm.N, ntm.M)
    ntm.read_vector = torch.zeros(ntm.M)

    for t in range(seq_len):
        _, state = ntm(test_seq[t], state)

    predicted = []
    blank = torch.zeros(input_size)
    for t in range(seq_len):
        out, state = ntm(blank, state)
        predicted.append((out > 0.5).int())

    print("Saída prevista:")
    print(torch.stack(predicted))
    print("Saída esperada:")
    print(target_seq.int())
