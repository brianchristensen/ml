import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

from synthesizer import Synthesizer
from node import Node
from router import Router
from gem import GEM

# Hyperparameters
num_epochs = 3
num_nodes = 1
max_ops = 1
max_gem = 10000
latent_dim = 128
symbolic_dim = latent_dim
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and pretrained T5 encoder
tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
t5_encoder = t5_model.get_encoder()
input_dim = t5_encoder.config.d_model  # typically 512
vocab_size = tokenizer.vocab_size

# Initialize Synthesizer components
nodes = [Node(latent_dim, symbolic_dim).to(device) for _ in range(num_nodes)]
router = Router(symbolic_dim, num_nodes).to(device)
gem = GEM(symbolic_dim, max_gem=max_gem, device=device)
synth = Synthesizer(nodes, router, gem, input_dim, latent_dim, device).to(device)

# Lightweight decoder head
decoder_head = nn.Linear(latent_dim, vocab_size).to(device)

# Optimizer
optimizer = optim.Adam(list(synth.parameters()) + list(decoder_head.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Load HotpotQA dataset
dataset = load_dataset('hotpot_qa', 'fullwiki', split='train[:1%]', trust_remote_code=True)

def collate_fn(batch):
    questions = [item['question'] for item in batch]
    contexts = []
    for item in batch:
        context_strings = []
        for para in item['context']:
            if isinstance(para, dict) and 'sentences' in para:
                context_strings.append(" ".join(para['sentences']))
        contexts.append(" ".join(context_strings))
    target_texts = [item['answer'] if 'answer' in item else item['answers'][0] for item in batch]

    # Tokenize inputs and targets
    inputs = tokenizer(
        [q + " " + c for q, c in zip(questions, contexts)],
        padding=True, truncation=True, return_tensors='pt'
    )
    labels = tokenizer(target_texts, padding=True, truncation=True, return_tensors='pt')

    return inputs, labels, questions, target_texts

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(num_epochs):
    synth.train()
    total_loss = 0
    for step, (inputs, labels, questions, target_answers) in enumerate(train_loader):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        label_ids = labels['input_ids'].to(device)

        # Encode with T5 encoder
        with torch.no_grad():
            encoder_outputs = t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden = encoder_outputs.last_hidden_state.mean(dim=1)  # [batch, dim]

        # Pass through Synthesizer
        latent, programs, sym_embeds = synth.forward(encoder_hidden, max_ops=max_ops)

        # Decoder head
        logits = decoder_head(latent)  # [batch, vocab_size]

        # Shift labels for decoder (standard seq2seq practice)
        target = label_ids[:, 1:].contiguous().view(-1)
        pred_logits = logits.view(-1, vocab_size)

        loss = loss_fn(pred_logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # Evaluation / generation
    synth.eval()
    with torch.no_grad():
        for inputs, labels, questions, target_answers in train_loader:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            encoder_outputs = t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden = encoder_outputs.last_hidden_state.mean(dim=1)

            latent, programs, sym_embeds = synth.forward(encoder_hidden, max_ops=max_ops)
            logits = decoder_head(latent)

            pred_ids = torch.argmax(logits, dim=-1)
            decoded_outputs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for q, target, pred in zip(questions, target_answers, decoded_outputs):
                print(f"\nQUESTION: {q}")
                print(f"TARGET ANSWER: {target}")
                print(f"GENERATED ANSWER: {pred}")
            break  # Only print one batch for sanity check after epoch

print("Training complete.")
