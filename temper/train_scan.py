import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime
from model import TemperNet

# --------- SCAN Dataset ---------
class ScanDataset(Dataset):
    def __init__(self, path, max_len=50):
        super().__init__()
        self.max_len = max_len
        self.examples = []
        

        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) % 2 == 0, "Expected IN/OUT pairs"

            for line in lines:
                assert "IN:" in line and "OUT:" in line, "Line must contain both IN and OUT"
                in_part, out_part = line.split("OUT:")
                cmd = in_part.replace("IN:", "").strip()
                act = out_part.strip()
                self.examples.append((cmd, act))

        self.command_vocab = self.build_vocab([cmd for cmd, _ in self.examples])
        self.action_vocab = self.build_vocab([act for _, act in self.examples])
        self.rev_action_vocab = {v: k for k, v in self.action_vocab.items()}

    def build_vocab(self, texts):
        vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        idx = 4
        for text in texts:
            for token in text.strip().split():
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    def encode(self, text, vocab, add_sos_eos=False):
        tokens = text.strip().split()
        if add_sos_eos:
            tokens = ["<SOS>"] + tokens + ["<EOS>"]
        token_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        if len(token_ids) < self.max_len:
            token_ids += [vocab["<PAD>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return torch.tensor(token_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        cmd, act = self.examples[idx]
        x = self.encode(cmd, self.command_vocab)
        y = self.encode(act, self.action_vocab, add_sos_eos=True)
        return x, y

# --------- Seq2Seq Wrapper ---------
class TemperSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tempers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.decoder_embedding = nn.Embedding(output_dim, hidden_dim)
        self.temper_net = TemperNet(hidden_dim, hidden_dim, output_dim, num_tempers=num_tempers)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, y_in):
        x_embed = self.embedding(x.long())
        x_rep = x_embed.mean(dim=1)
        z = self.temper_net(x_rep, return_hidden=True)

        y_embed = self.decoder_embedding(y_in.long())
        z = z.unsqueeze(1).repeat(1, y_embed.size(1), 1)
        decoder_input = y_embed + z
        h0 = torch.zeros(1, x.size(0), z.size(-1), device=z.device)

        dec_out, _ = self.decoder(decoder_input, h0)
        logits = self.output_proj(dec_out)
        return logits

    def greedy_decode(self, x, max_len, start_token_id, eos_token_id):
        self.eval()  # Set eval mode just in case
        with torch.no_grad():
            x_embed = self.embedding(x.long())
            x_rep = x_embed.mean(dim=1)
            z = self.temper_net(x_rep, return_hidden=True)

            batch_size = x.size(0)
            outputs = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=x.device)
            h = torch.zeros(1, batch_size, z.size(-1), device=x.device)

            finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

            for _ in range(max_len):
                y_embed = self.decoder_embedding(outputs[:, -1])  # (B, D)
                y_embed = y_embed.unsqueeze(1)  # (B, 1, D)
                z_repeated = z.unsqueeze(1)
                decoder_input = y_embed + z_repeated

                dec_out, h = self.decoder(decoder_input, h)
                logits = self.output_proj(dec_out[:, -1])  # (B, V)
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

                outputs = torch.cat([outputs, next_token], dim=1)

                finished |= (next_token.squeeze(1) == eos_token_id)
                if finished.all():
                    break

            return outputs[:, 1:]  # Remove initial <SOS>

    def update_tempers_with_local_rewards(self, y_pred, y_target):
        return self.temper_net.update_tempers_with_local_rewards(y_pred, y_target)

    def print_epoch_summary(self, epoch, loss):
        self.temper_net.print_epoch_summary(epoch, loss)

    def print_routing_diagnostics(self):
        self.temper_net.print_routing_diagnostics()

    def reset_batch(self):
        self.temper_net.reset_batch()

    def reset_epoch(self):
        self.temper_net.reset_epoch()

    def dump_routing_summary(self, path):
        self.temper_net.dump_routing_summary(path)

# --------- Training Loop ---------
def train():
    path_to_data = "data/scan.txt"
    dataset = ScanDataset(path_to_data, max_len=20)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = len(dataset.command_vocab)
    output_dim = len(dataset.action_vocab)

    epochs = 20
    hidden_dim = 8
    num_tempers = 4

    model = TemperSeq2Seq(input_dim, hidden_dim, output_dim, num_tempers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.action_vocab["<PAD>"])

    start_time = time.time()
    print(f"\U0001f9e0 Training Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(epochs+1):
        epoch_start_time = time.time()
        total_task_loss = total_policy_loss = 0
        correct_tf = 0  # Teacher-forced accuracy
        correct_greedy = 0  # Greedy decoding accuracy
        total = 0

        model.train()
        for x, y in loader:
            y_input = y[:, :-1]
            y_target = y[:, 1:]

            optimizer.zero_grad()
            y_pred = model(x, y_input)
            task_loss = criterion(y_pred.reshape(-1, y_pred.size(-1)), y_target.reshape(-1))
            policy_loss = model.update_tempers_with_local_rewards(y_pred, y_target)
            loss = task_loss + policy_loss
            loss.backward()
            optimizer.step()
            
            total_task_loss += task_loss.item()
            total_policy_loss += policy_loss.item()
            model.reset_batch()

            # === Evaluate teacher-forced predictions ===
            y_pred_ids = y_pred.argmax(dim=-1)
            for i in range(y_pred_ids.size(0)):
                pred = [t.item() for t in y_pred_ids[i] if t != dataset.action_vocab["<PAD>"]]
                target = [t.item() for t in y_target[i] if t != dataset.action_vocab["<PAD>"]]
                if pred == target:
                    correct_tf += 1
                total += 1

        # === Evaluate greedy decoding on the full training set ===
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                y_target = y[:, 1:]
                decoded = model.greedy_decode(
                    x,
                    max_len=y_target.size(1),
                    start_token_id=dataset.action_vocab["<SOS>"],
                    eos_token_id=dataset.action_vocab["<EOS>"]
                )
                for i in range(decoded.size(0)):
                    pred = [t.item() for t in decoded[i] if t.item() not in [dataset.action_vocab["<PAD>"], dataset.action_vocab["<EOS>"]]]
                    target = [t.item() for t in y_target[i] if t.item() not in [dataset.action_vocab["<PAD>"], dataset.action_vocab["<EOS>"]]]
                    if pred == target:
                        correct_greedy += 1

        tf_acc = 100 * correct_tf / total
        greedy_acc = 100 * correct_greedy / total
        print(f"\nEpoch {epoch}:")
        print(f"  Teacher-Forced Accuracy: {tf_acc:.2f}%")
        print(f"  Greedy Decoding Accuracy: {greedy_acc:.2f}%")

        # Print one example decoded sequence
        if epoch % 5 == 0:
            example_x, example_y = next(iter(loader))
            decoded = model.greedy_decode(
                example_x,
                max_len=dataset.max_len,
                start_token_id=dataset.action_vocab["<SOS>"],
                eos_token_id=dataset.action_vocab["<EOS>"]
            )
            idx = 0
            pred_tokens = [
                dataset.rev_action_vocab[t.item()]
                for t in decoded[idx] if t.item() not in [dataset.action_vocab["<PAD>"], dataset.action_vocab["<EOS>"]]
            ]
            target_tokens = [
                dataset.rev_action_vocab[t.item()]
                for t in example_y[idx][1:] if t.item() not in [dataset.action_vocab["<PAD>"], dataset.action_vocab["<EOS>"]]
            ]
            print("Example Prediction:")
            print("Predicted:", " ".join(pred_tokens))
            print("Target:   ", " ".join(target_tokens))
            epoch_duration = time.time() - epoch_start_time

        model.print_epoch_summary(epoch, total_task_loss, epoch_duration)
        model.print_routing_diagnostics()
        model.reset_epoch()

    model.dump_routing_summary("logs/scan_routing_summary.csv")

if __name__ == "__main__":
    train()
