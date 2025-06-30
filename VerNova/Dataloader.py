import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

PAD_TOKEN_ID = 999
EOS_TOKEN_ID = 998  # novo token de fim

vocab = {
    "<PAD>": PAD_TOKEN_ID,
    "<EOS>": EOS_TOKEN_ID,
    "q0": 0, "HALT": 1,
    "0": 2, "1": 3, "_": 4,
    "L": 5, "R": 6
}

class BinaryCopyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_seq = item["input"] + [EOS_TOKEN_ID]
        output_seq = item["output"] + [EOS_TOKEN_ID]  # achata e adiciona <EOS>
        return {
            "input": torch.tensor(input_seq, dtype=torch.long),
            "output": torch.tensor(output_seq, dtype=torch.long)
        }

def collate_fn(batch):
    input_seqs = [item["input"] for item in batch]
    output_seqs = [item["output"] for item in batch]

    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=PAD_TOKEN_ID)
    output_padded = pad_sequence(output_seqs, batch_first=True, padding_value=PAD_TOKEN_ID)

    return {
        "input": input_padded,
        "output": output_padded
    }

if __name__ == "__main__":
    dataset_path = "dataset.json"
    batch_size = 8

    dataset = BinaryCopyDataset(jsonl_path=dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for batch in loader:
        print("Input shape: ", batch["input"].shape)
        print("Output shape:", batch["output"].shape)
        print("Exemplo de entrada:", batch["input"][0])
        print("Exemplo de saída: ", batch["output"][0])
        break
