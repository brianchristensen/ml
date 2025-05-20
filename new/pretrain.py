import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from model import CognitionModel

# --- Configuration ---
HIDDEN_DIM = 128
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = "models/pretrained_model.pth"
TOKENIZER_NAME = "bert-base-uncased"

# --- MLM Dataset ---
class MLMDataset(Dataset):
    def __init__(self, tokenizer, texts, model, embedding, mask_prob=0.15, max_length=256):
        self.tokenizer = tokenizer
        self.texts = [t for t in texts if len(t.strip()) > 0]
        self.mask_prob = mask_prob
        self.max_length = max_length
        self.model = model
        self.embedding = embedding
        self.model.eval()  # Set model to eval for concept predictions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        labels = input_ids.clone()

        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Embed and pass through model to get token_type and polarity
        with torch.no_grad():
            device = self.embedding.weight.device
            input_ids_gpu = input_ids.unsqueeze(0).to(device)
            attention_mask_gpu = attention_mask.unsqueeze(0).to(device)

            x = self.embedding(input_ids_gpu)  # (1, T, D)
            _, _, _, concepts = self.model(x, attention_mask=attention_mask_gpu)
            focus_logits = concepts["question_focus"][0]       # (T, 5)
            role_logits = concepts["answer_role"][0]           # (T, 4)
            negation_logits = concepts["negation_scope"][0]    # (T, 2)

            question_focus = torch.argmax(focus_logits, dim=-1)      # (T,)
            answer_role = torch.argmax(role_logits, dim=-1)          # (T,)
            negation_scope = torch.argmax(negation_logits, dim=-1)   # (T,)

        # Create mask: mask nouns (token_type==1), verbs (==2), and positive polarity (==1)
        pad_id = torch.tensor(self.tokenizer.pad_token_id, device=input_ids.device)
        cls_id = torch.tensor(self.tokenizer.cls_token_id, device=input_ids.device)
        sep_id = torch.tensor(self.tokenizer.sep_token_id, device=input_ids.device)

        mask_arr = (
            (
                (answer_role >= 1) |           # clue, supports, contradicts
                (question_focus == 4) |        # yes/no questions
                (negation_scope == 1)          # in negation
            ) &
            (input_ids != pad_id) &
            (input_ids != cls_id) &
            (input_ids != sep_id)
        )
        # Apply random sampling to avoid full masking
        rand = torch.rand(input_ids.shape, device=input_ids.device)
        mask_arr = mask_arr & (rand < self.mask_prob)

        input_ids[mask_arr] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- MLM Head ---
class MLMHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):  # (B, T, D)
        return self.mlp(x)

# --- Main Pretraining Routine ---
def pretrain():
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    print("Loading QA dataset...")
    # nq = load_dataset("natural_questions", split="train[:50000]")  # Trim for speed

    # texts = []
    # for ex in nq:
    #     try:
    #         question = ex['question']['text']
    #         tokens = ex.get('document', {}).get('tokens', [])
    #         if not tokens or len(tokens) < 10:
    #             continue
    #         context = " ".join([t['token'] for t in tokens if not t.get('html_token', False)])
    #         context = context.strip()[:512]
    #         if question and context:
    #             texts.append(f"Q: {question} C: {context}")
    #     except Exception as e:
    #         continue

    # print(f"✅ Loaded {len(texts)} QA examples.")

    trivia = load_dataset("trivia_qa", "unfiltered.nocontext", split="train[:10000]")
    texts = [
        f"Q: {ex['question']} A: {ex['answer']['value']}"
        for ex in trivia
        if ex['answer'] and ex['answer']['value']
    ]
    print(f"Loaded {len(texts)} TriviaQA examples.")


    print("Initializing model...")
    model = CognitionModel(hidden_dim=HIDDEN_DIM).to(DEVICE)
    embedding = nn.Embedding(tokenizer.vocab_size, HIDDEN_DIM).to(DEVICE)

    mlm_dataset = MLMDataset(tokenizer, texts, model, embedding, max_length=MAX_LEN)
    mlm_loader = DataLoader(mlm_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load BoolQ
    boolq_raw = load_dataset("boolq")["train"]
    class BoolQDataset(Dataset):
        def __init__(self, tokenizer, data, max_length=256):
            self.tokenizer = tokenizer
            self.data = data
            self.max_length = max_length
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = tokenizer(
                item["question"], item["passage"],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "label": torch.tensor(item["answer"], dtype=torch.long)
            }
    boolq_dataset = BoolQDataset(tokenizer, boolq_raw, max_length=MAX_LEN)
    boolq_loader = DataLoader(boolq_dataset, batch_size=BATCH_SIZE, shuffle=True)
    boolq_iter = iter(boolq_loader)

    mlm_head = MLMHead(HIDDEN_DIM, tokenizer.vocab_size).to(DEVICE)
    classifier_head = nn.Linear(HIDDEN_DIM, 2).to(DEVICE)

    optimizer = optim.Adam(
        list(model.parameters()) + 
        list(mlm_head.parameters()) +
        list(classifier_head.parameters()), lr=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print("Starting multi-task pretraining...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        mlm_head.train()
        classifier_head.train()
        total_loss = 0

        for i, batch in enumerate(mlm_loader):
            # MLM batch
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            embedded = embedding(input_ids)
            pooled, symbolic_out, symbolic_entropy, concepts = model(embedded, attention_mask=attention_mask)
            logits = mlm_head(symbolic_out)
            mlm_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Try to fetch BoolQ batch
            try:
                boolq_batch = next(boolq_iter)
            except StopIteration:
                boolq_iter = iter(boolq_loader)
                boolq_batch = next(boolq_iter)

            b_input_ids = boolq_batch['input_ids'].to(DEVICE)
            b_attention_mask = boolq_batch['attention_mask'].to(DEVICE)
            b_labels = boolq_batch['label'].to(DEVICE)

            b_embedded = embedding(b_input_ids)
            b_pooled, _, symbolic_entropy, _ = model(b_embedded, attention_mask=b_attention_mask)
            boolq_logits = classifier_head(b_pooled)
            boolq_loss = criterion(boolq_logits, b_labels)

            # Combine losses
            total_batch_loss = (
                mlm_loss + 
                0.1 * -symbolic_entropy +
                0.1 * boolq_loss
            )

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(mlm_loader)
        print(f"[Pretrain Epoch {epoch}] Total Multi-task Loss: {avg_loss:.4f}")

    print(f"Saving pretrained model to: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    pretrain()
