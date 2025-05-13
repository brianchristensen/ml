# boolq.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer
from model import BusSynthesizer

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, z):
        return self.mlp(z)
    
class BoolQDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, max_length=256):
        self.dataset = load_dataset('boolq', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        passage = item['passage']
        question = item['question']
        label = int(item['answer'])  # bool → int

        encoding = self.tokenizer(
            question,
            passage,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

class BoolQTrainer:
    def __init__(self, model, classifier, tokenizer, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.classifier = classifier.to(device)
        self.tokenizer = tokenizer

        self.train_dataset = BoolQDataset('train', tokenizer)
        self.test_dataset = BoolQDataset('validation', tokenizer)

        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()), lr=1e-4
        )

        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.model.latent_dim).to(self.device)

    def train_epoch(self):
        self.model.train()
        self.classifier.train()
        total_loss, total_correct = 0, 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            embedded = self.embedding(input_ids)

            token_outputs = self.model(embedded)
            pooled = token_outputs.mean(dim=1)

            logits = self.classifier(pooled)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_acc = total_correct / len(self.train_loader.dataset)
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")

    def evaluate(self):
        self.model.eval()
        self.classifier.eval()
        total_loss, total_correct = 0, 0
        program_counter = Counter()
        halt_lengths = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                embedded = self.embedding(input_ids)

                token_outputs, program = self.model(embedded, return_program=True)
                pooled = token_outputs.mean(dim=1)
                logits = self.classifier(pooled)
                loss = self.criterion(logits, labels)

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item() * input_ids.size(0)

                # === Routing trace ===
                B, S, H = program.shape
                halt_index = self.model.num_nodes  # last index is HALT

                flat_programs = program.view(-1, program.size(-1)).cpu().tolist()
                for prog in flat_programs:
                    if halt_index in prog:
                        halt_pos = prog.index(halt_index)
                        prog = prog[:halt_pos]
                    program_counter[tuple(prog)] += 1
                    halt_lengths.append(len(prog))

        stds = model.token_prompts[0, :S].std(dim=1)
        print(f"[Token Prompt Std] mean: {stds.mean().item():.4f}, max: {stds.max().item():.4f}")

        # === Program usage summary ===
        print(f"\nTop 10 symbolic programs:")
        for prog, count in program_counter.most_common(10):
            print(f"  {prog} used {count} times")

        avg_len = sum(halt_lengths) / len(halt_lengths)
        print(f"Avg halt steps: {avg_len:.2f}")

        avg_loss = total_loss / len(self.test_loader.dataset)
        avg_acc = total_correct / len(self.test_loader.dataset)
        print(f"Test Loss: {avg_loss:.4f}, Test Acc: {avg_acc:.4f}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 10
    num_nodes = 4
    max_ops = 4
    num_classes = 2
    input_dim = 128
    latent_dim = 128
    symbolic_dim = 128

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = BusSynthesizer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        symbolic_dim=symbolic_dim,
        num_nodes=num_nodes,
        max_ops=max_ops
    ).to(device)

    classifier = ClassifierHead(latent_dim, num_classes)
    trainer = BoolQTrainer(model, classifier, tokenizer, device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}")
        trainer.train_epoch()
        trainer.evaluate()
