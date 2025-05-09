import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import collections
import string
from synthesizer import Synthesizer
from node import Node
from router import Router
from gem import GEM

# Hyperparameters
num_epochs = 20
num_nodes = 8
max_ops = 5
max_gem = 10000
latent_dim = 128
symbolic_dim = 512
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and pretrained T5 model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
t5_encoder = t5_model.get_encoder()
input_dim = t5_encoder.config.d_model  # typically 512

# Initialize Synthesizer components
nodes = [Node(latent_dim, symbolic_dim).to(device) for _ in range(num_nodes)]
router = Router(latent_dim, symbolic_dim, num_nodes).to(device)
gem = GEM(symbolic_dim, max_gem=max_gem, device=device)
synth = Synthesizer(nodes, router, gem, input_dim, latent_dim, device).to(device)

# Adapter to map Synth latent → T5 encoder embedding space
class SynthToEncoderAdapter(nn.Module):
    def __init__(self, latent_dim, encoder_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, encoder_dim)
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, latent, seq_len):
        adapted = self.norm(self.proj(latent))  # [batch, encoder_dim]
        expanded = adapted.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, encoder_dim]
        return expanded

adapter = SynthToEncoderAdapter(latent_dim, t5_model.config.d_model).to(device)

# Optimizer (Synth + adapter + T5 decoder)
optimizer = optim.Adam(
    list(synth.parameters()) + list(adapter.parameters()) + list(t5_model.decoder.parameters()),
    lr=1e-4
)
loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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

def clean_and_tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

def compute_token_f1(prediction, ground_truth):
    pred_tokens = clean_and_tokenize(prediction)
    gt_tokens = clean_and_tokenize(ground_truth)
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(num_epochs):
    synth.train()
    adapter.train()
    total_loss = 0
    reward_list = []

    for step, (inputs, labels, questions, target_answers) in enumerate(train_loader):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        label_ids = labels['input_ids'].to(device)

        # Encode with T5 encoder (no gradients)
        with torch.no_grad():
            encoder_outputs = t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden = encoder_outputs.last_hidden_state.mean(dim=1)  # [batch, dim]

        # Synthesizer forward
        latent, programs, sym_embeds = synth.forward(encoder_hidden, max_ops=max_ops)  # [batch, latent_dim]

        # Adapt and expand Synth latent to match input sequence
        latent_embeds = adapter(latent, seq_len=input_ids.size(1))  # [batch, seq_len, encoder_dim]

        # Prepare decoder inputs and shifted labels
        decoder_input_ids = label_ids[:, :-1].contiguous()
        decoder_labels = label_ids[:, 1:].contiguous()

        outputs = t5_model(
            inputs_embeds=latent_embeds,
            decoder_input_ids=decoder_input_ids,
            labels=None  # we compute loss externally
        )

        lm_logits = outputs.logits  # [batch, decoder_seq_len, vocab_size]
        
        # Ensure sequence length alignment
        assert lm_logits.shape[1] == decoder_labels.shape[1], \
            f"Logits length {lm_logits.shape[1]} vs labels length {decoder_labels.shape[1]} mismatch"

        # Flatten for loss
        loss = loss_fct(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            decoder_labels.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            generated_ids = t5_model.generate(inputs_embeds=latent_embeds, max_length=64)
            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, target in zip(decoded_outputs, target_answers):
                pred_clean = pred.strip()
                if pred_clean:
                    reward = compute_token_f1(pred_clean, target)
                    if reward != 0.0: print(f"Tokenf1 reward: {reward}")
                else:
                    reward = 0.0
                reward_list.append(torch.tensor([reward], device=device))

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # Update GEM with all accumulated rewards
    all_rewards = torch.cat(reward_list, dim=0)
    print(f"Collected programs: {len(synth.collected_programs)}")
    print(f"Collected rewards: {all_rewards.size(0)}")
    synth.update_gem(all_rewards)
     
    # Evaluation (single batch sanity check)
    synth.eval()
    adapter.eval()
    with torch.no_grad():
        for inputs, labels, questions, target_answers in train_loader:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            encoder_outputs = t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden = encoder_outputs.last_hidden_state.mean(dim=1)

            latent, programs, sym_embeds = synth.forward(encoder_hidden, max_ops=max_ops)
            latent_embeds = adapter(latent, seq_len=input_ids.size(1))

            generated_ids = t5_model.generate(inputs_embeds=latent_embeds, max_length=64)
            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for q, target, pred in zip(questions, target_answers, decoded_outputs):
                print(f"\nQUESTION: {q}")
                print(f"TARGET ANSWER: {target}")
                print(f"GENERATED ANSWER: {pred}")
            break  # Only one batch

    gem.print_top_n(10)

print("Training complete.")
