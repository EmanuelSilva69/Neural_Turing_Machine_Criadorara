import random
import json
import math
import itertools

# Vocabulário fixo
vocab = {
    "<PAD>": 999,
    "<EOS>": 998,
    "q0": 0, "HALT": 1,
    "0": 2, "1": 3, "_": 4,
    "L": 5, "R": 6
}
inv_vocab = {v: k for k, v in vocab.items()}

def tokenize_rule(state, read, write, move, next_state):
    def safe(s): return vocab[s] if s in vocab else vocab.setdefault(s, len(vocab))
    return [safe(state), safe(read), safe(write), safe(move), safe(next_state)]

def generate_mt_for_sequence(binary_seq):
    rules = []
    seq_len = len(binary_seq)

    for i in range(seq_len):
        state = f"q{i}"
        next_state = f"q{i+1}"
        read = str(binary_seq[i])
        rules.append(tokenize_rule(state, read, read, "R", next_state))

    state = f"q{seq_len}"
    rules.append(tokenize_rule(state, "_", "_", "R", f"q{seq_len+1}"))

    for i, bit in enumerate(binary_seq):
        state = f"q{seq_len+1+i}"
        next_state = f"q{seq_len+1+i+1}"
        rules.append(tokenize_rule(state, "_", str(bit), "R", next_state))

    rules.append(tokenize_rule(f"q{seq_len*2+1}", "_", "_", "R", "HALT"))
    rules.append([vocab["<EOS>"]])  # <EOS> token

    return rules

def generate_all_combinations_for_length(bit_len):
    all_sequences = list(itertools.product([0, 1], repeat=bit_len))
    random.shuffle(all_sequences)
    return all_sequences

def pad_sequence(seq, max_len):
    return seq + [vocab["<PAD>"]] * (max_len - len(seq))

def generate_curriculum_dataset(
    total_examples=15000,
    min_len=2,
    max_len=16,
    output_file="dataset.json",
    balanced_curriculum=False
):
    dataset = []
    total_generated = 0

    if not balanced_curriculum:
        for bit_len in range(min_len, max_len + 1):
            max_combinations = 2 ** bit_len
            remaining = total_examples - total_generated
            if remaining <= 0:
                break
            to_generate = min(max_combinations, remaining)

            print(f"[+] Gerando {to_generate} exemplos com {bit_len} bits")
            sequences = generate_all_combinations_for_length(bit_len)[:to_generate]

            for seq in sequences:
                input_seq = list(seq) + [vocab["<EOS>"]]
                output_seq = generate_mt_for_sequence(seq)
                dataset.append({
                    "input": input_seq,
                    "output": list(itertools.chain.from_iterable(output_seq))
                })

            total_generated += to_generate
    else:
        bit_len_limits = {
            2: 4,
            3: 8,
            4: 16,
            5: 32,
            6: 64,
            7: 128,
            8: 256,
            9: 512,
            10: 1024,
            11: 2048,
            12: 2048,
            13: 2048,
            14: 2048,
            15: 2048,
            16: 2048,
        }

        for bit_len in range(min_len, max_len + 1):
            limit = bit_len_limits.get(bit_len, 1024)
            remaining = total_examples - total_generated
            if remaining <= 0:
                break
            to_generate = min(limit, remaining)
            print(f"[+] Gerando {to_generate} exemplos com {bit_len} bits")
            sequences = generate_all_combinations_for_length(bit_len)[:to_generate]

            for seq in sequences:
                input_seq = list(seq) + [vocab["<EOS>"]]
                output_seq = generate_mt_for_sequence(seq)
                dataset.append({
                    "input": input_seq,
                    "output": list(itertools.chain.from_iterable(output_seq))
                })

            total_generated += to_generate

    max_input_len = max(len(ex["input"]) for ex in dataset)
    max_output_len = max(len(ex["output"]) for ex in dataset)

    for ex in dataset:
        ex["input"] = pad_sequence(ex["input"], max_input_len)
        ex["output"] = pad_sequence(ex["output"], max_output_len)

    random.shuffle(dataset)

    with open(output_file, "w") as f:
        for example in dataset:
            json.dump(example, f, separators=(",", ":"))
            f.write("\n")

    print(f"\n✅ Dataset curriculum salvo em '{output_file}' com {len(dataset)} exemplos.")

if __name__ == "__main__":
    generate_curriculum_dataset(
        total_examples=8000,
        min_len=2,
        max_len=16,
        output_file="dataset.json",
        balanced_curriculum=True
    )
