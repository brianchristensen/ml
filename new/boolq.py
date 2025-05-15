# boolq.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import time
from transformers import AutoTokenizer
from model import CognitionModel

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_codes(model):
    codebook_np = model.system2.concept_graph.codebook.detach().cpu().numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(codebook_np)
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.title("VQ Codebook Embeddings")
    plt.show() # Add plt.show() to display the plot

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

        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.model.hidden_dim).to(self.device)

    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        self.model.train()
        self.classifier.train()
        total_loss, total_correct = 0, 0
    
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            embedded = self.embedding(input_ids)

            pooled = self.model(embedded)

            logits = self.classifier(pooled)
            task_loss = self.criterion(logits, labels)
            loss = task_loss + self.model.system2.concept_graph.vq_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_acc = total_correct / len(self.train_loader.dataset)
        epoch_duration = time.time() - epoch_start_time
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Duration: {epoch_duration}s")

    def evaluate(self, epoch):
        self.model.eval()
        self.classifier.eval()
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                embedded = self.embedding(input_ids)

                pooled = self.model(embedded, return_routing_trace=True)
                logits = self.classifier(pooled)
                task_loss = self.criterion(logits, labels)
                loss = task_loss + self.model.system2.concept_graph.vq_loss

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item() * input_ids.size(0)


        avg_loss = total_loss / len(self.test_loader.dataset)
        avg_acc = total_correct / len(self.test_loader.dataset)
        print(f"Test Loss: {avg_loss:.4f}, Test Acc: {avg_acc:.4f}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 10
    num_classes = 2
    hidden_dim = 128

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = CognitionModel(hidden_dim)

    classifier = ClassifierHead(hidden_dim, num_classes)
    trainer = BoolQTrainer(model, classifier, tokenizer, device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}")
        trainer.train_epoch(epoch + 1)
        trainer.evaluate(epoch + 1)
        #plot_codes(model)
