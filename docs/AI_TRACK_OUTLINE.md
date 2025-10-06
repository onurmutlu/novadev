# 🧠 NovaDev AI Mastery — Week 1–8 Master Track

> **"Matematikten Modele, Modelden Ürüne"** — Baron 🔥

**Program:** NovaDev v1.1 — AI Hattı  
**Seviye:** Beginner to Advanced  
**Süre:** 8 hafta (80-100 saat toplam)  
**Format:** T → P → X (Teori → Pratik → Ürün)

---

## 🎯 Program Hedefi

8 hafta sonunda:
- ✅ ML fundamentals (sıfırdan implementation)
- ✅ Deep Learning (MLP → CNN → Transformer)
- ✅ NLP (Türkçe BERT fine-tuning)
- ✅ RAG Pipeline (dokümandan yanıt)
- ✅ Tool-calling Agent (multi-step reasoning)
- ✅ LoRA fine-tuning (özelleştirilmiş LLM)
- ✅ Production API (FastAPI + Docker)

**Çıktı:** Kendi AI modellerini sıfırdan yazabilen, fine-tune edebilen, production'a deploy edebilen **complete AI engineer**.

---

## 📚 İçindekiler

- [Week 0: Foundations (Complete ✅)](#week-0-foundations-complete-)
- [Week 1: Linear Regression](#week-1--linear-regression)
- [Week 2: MLP & MNIST](#week-2--mlp--mnist)
- [Week 3: NLP & Türkçe BERT](#week-3--nlp--türkçe-bert)
- [Week 4: RAG Pipeline](#week-4--rag-pipeline)
- [Week 5: Tool-calling Agent](#week-5--tool-calling-agent)
- [Week 6: LoRA Fine-tuning](#week-6--lora-fine-tuning)
- [Week 7: Service & Monitoring](#week-7--service--monitoring)
- [Week 8: Capstone E2E](#week-8--capstone-e2e)
- [Sertifikasyon](#-sertifikasyon--novabaron-ai-l1)

---

## Week 0: Foundations (Complete ✅)

**Status:** COMPLETE (7,061 satır teori + setup)

**Achievements:**
- ✅ ML Fundamentals (theory_foundations.md)
- ✅ Mathematical Foundations (theory_mathematical.md)
- ✅ PyTorch setup & MPS verification
- ✅ First tensor operations (hello_tensor.py)
- ✅ Self-assessment complete

**Deliverables:**
- `week0_setup/` (7 documents, 5 levels)
- `theory_closure.md` (self-check)
- Environment verified (Python 3.11+, PyTorch, MPS)

---

## Week 1 — 📘 Linear Regression

**Hedef:** Linear regression'ı sıfırdan implement etmek, train/val split, early stopping, L2 regularization.

### Kazanımlar

**Teori (T):**
- Linear model: f(x) = W·x + b
- Loss functions: MSE, MAE, Huber
- Gradient descent variants (SGD, Momentum, Adam)
- Regularization: L1, L2, Elastic Net
- Train/Val/Test split
- Early stopping & overfitting detection
- Hyperparameter tuning

**Pratik (P):**
- Manual gradient descent implementation
- PyTorch nn.Module version
- Data loading & preprocessing
- Training loop with validation
- Learning rate scheduling
- Ablation studies

**Ürün (X):**
- Production-ready training script
- Model checkpointing
- Experiment logging
- Metrics visualization
- Val MSE ≤ 0.50 target

### Görevler

```bash
# 1. Manual implementation
week1_tensors/
├── linreg_manual.py       # Manual GD from scratch
├── linreg_module.py       # nn.Module version
├── data_utils.py          # Dataset & DataLoader
└── train.py               # Training script

# 2. Experiments
week1_tensors/experiments/
├── lr_sweep.py            # Learning rate sweep
├── ablation.py            # Ablation study
└── regularization.py      # L2 comparison

# 3. Outputs
outputs/ai/week1/
├── final_best_model.pt
├── training_curves.png
├── ablation_results.png
└── metrics.json
```

### Training Script Example

```python
# week1_tensors/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_with_early_stopping(
    model, train_loader, val_loader, 
    epochs=200, lr=0.01, l2=0.001, patience=20
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'outputs/ai/week1/best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    
    return train_losses, val_losses, best_val_loss

# Usage
if __name__ == "__main__":
    # Generate synthetic data
    X_train = torch.randn(800, 10)
    y_train = X_train @ torch.randn(10, 1) + torch.randn(800, 1) * 0.1
    X_val = torch.randn(200, 10)
    y_val = X_val @ torch.randn(10, 1) + torch.randn(200, 1) * 0.1
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = LinearRegression(input_dim=10)
    train_losses, val_losses, best_val = train_with_early_stopping(
        model, train_loader, val_loader, epochs=200, lr=0.01, l2=0.001
    )
    
    print(f"Best Val MSE: {best_val:.4f}")
```

### DoD (Definition of Done)

- [ ] Manual GD implementation working
- [ ] nn.Module version with same results
- [ ] Train/Val split (80/20)
- [ ] Early stopping implemented
- [ ] L2 regularization working
- [ ] Val MSE ≤ 0.50 achieved
- [ ] Training curves plotted
- [ ] LR sweep (0.001, 0.01, 0.1) completed
- [ ] Ablation study (no L2, no early stop)
- [ ] Final report written

### Çıktılar

```
reports/
└── ai/
    └── w1_summary.md

outputs/
└── ai/
    └── week1/
        ├── final_best_model.pt
        ├── training_curves.png
        ├── lr_sweep.png
        ├── ablation_results.png
        └── metrics.json
```

---

## Week 2 — 📗 MLP & MNIST

**Hedef:** Multi-Layer Perceptron ile MNIST digit classification, 97%+ accuracy.

### Kazanımlar

**Teori (T):**
- Neural network architecture
- Activation functions (ReLU, Tanh, Sigmoid)
- Hidden layers & depth
- Classification loss: Cross-Entropy
- Softmax & probabilities
- Evaluation metrics: Accuracy, Precision, Recall, F1

**Pratik (P):**
- MLP implementation (2-3 hidden layers)
- MNIST dataset loading
- Batch normalization
- Dropout regularization
- Learning rate scheduling
- Confusion matrix analysis

**Ürün (X):**
- MNIST classifier (97%+ accuracy)
- Model visualization
- Inference script
- Error analysis

### Görevler

```bash
# 1. MLP implementation
week2_mlp/
├── mlp.py                 # MLP architecture
├── train.py               # Training script
├── evaluate.py            # Evaluation & metrics
└── infer.py               # Single image inference

# 2. Experiments
week2_mlp/experiments/
├── depth_ablation.py      # 1, 2, 3 hidden layers
├── activation_compare.py  # ReLU vs Tanh
└── regularization.py      # Dropout sweep

# 3. Outputs
outputs/ai/week2/
├── mlp_best.pt
├── confusion_matrix.png
├── training_curves.png
└── error_analysis.png
```

### MLP Architecture

```python
# week2_mlp/mlp.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128], num_classes=10, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)
```

### DoD (Definition of Done)

- [ ] MLP trained on MNIST
- [ ] Test accuracy ≥ 97%
- [ ] Confusion matrix generated
- [ ] Error analysis (top 10 misclassified)
- [ ] Depth ablation (1, 2, 3 layers)
- [ ] Activation comparison (ReLU vs Tanh)
- [ ] Dropout sweep (0.0, 0.2, 0.5)
- [ ] Inference script tested
- [ ] Model saved & loadable

### Çıktılar

```
reports/
└── ai/
    └── w2_mnist.md

outputs/
└── ai/
    └── week2/
        ├── mlp_best.pt
        ├── confusion_matrix.png
        ├── training_curves.png
        ├── error_analysis.png
        └── metrics.json
```

---

## Week 3 — 📘 NLP & Türkçe BERT

**Hedef:** Türkçe sentiment analysis, BERT fine-tuning, F1 ≥ 0.85.

### Kazanımlar

**Teori (T):**
- Tokenization (WordPiece, SentencePiece)
- Word embeddings (Word2Vec, GloVe)
- Transformer architecture basics
- Attention mechanism intuition
- Pre-trained models (BERT, GPT)
- Fine-tuning vs feature extraction
- Turkish NLP challenges

**Pratik (P):**
- HuggingFace Transformers library
- BERT fine-tuning for classification
- Turkish dataset preprocessing
- Training with mixed precision
- Hyperparameter optimization
- Error analysis

**Ürün (X):**
- Turkish sentiment classifier
- Fine-tuned BERT model
- Inference API endpoint
- Evaluation report

### Görevler

```bash
# 1. NLP pipeline
week3_nlp/
├── dataset.py             # Turkish dataset loader
├── tokenizer_utils.py     # Tokenization helpers
├── finetune_bert.py       # BERT fine-tuning script
├── evaluate.py            # Metrics & confusion matrix
└── infer.py               # Inference script

# 2. Experiments
week3_nlp/experiments/
├── lr_schedule.py         # LR scheduler comparison
├── batch_size.py          # Batch size ablation
└── freeze_layers.py       # Layer freezing study

# 3. Outputs
outputs/ai/week3/
├── bert_turkish_sentiment/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer_config.json
└── metrics/
    ├── confusion_matrix.png
    ├── f1_scores.png
    └── error_analysis.csv
```

### BERT Fine-tuning Script

```python
# week3_nlp/finetune_bert.py
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def main():
    # Load Turkish BERT
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3  # pos, neg, neutral
    )
    
    # Load dataset (example: Turkish product reviews)
    dataset = load_dataset("turkish_product_reviews", split="train")
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True,
            max_length=128
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="outputs/ai/week3/bert_turkish_sentiment",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,  # Mixed precision
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model("outputs/ai/week3/bert_turkish_sentiment")
    tokenizer.save_pretrained("outputs/ai/week3/bert_turkish_sentiment")

if __name__ == "__main__":
    main()
```

### DoD (Definition of Done)

- [ ] BERT fine-tuned on Turkish dataset
- [ ] Test F1 ≥ 0.85
- [ ] Confusion matrix generated
- [ ] Error analysis (20+ examples)
- [ ] LR schedule comparison (linear vs cosine)
- [ ] Batch size ablation (8, 16, 32)
- [ ] Layer freezing study (freeze first N layers)
- [ ] Inference script tested
- [ ] Model saved to HuggingFace format

### Çıktılar

```
reports/
└── ai/
    └── w3_nlp.md

outputs/
└── ai/
    └── week3/
        ├── bert_turkish_sentiment/
        │   ├── pytorch_model.bin
        │   ├── config.json
        │   └── tokenizer_config.json
        └── metrics/
            ├── confusion_matrix.png
            ├── f1_scores.png
            └── error_analysis.csv
```

---

## Week 4 — 📗 RAG Pipeline

**Hedef:** Retrieval-Augmented Generation sistemi kurmak, doküman RAG ile Recall@k ≥ 60%.

### Kazanımlar

**Teori (T):**
- RAG architecture (retriever + generator)
- Vector databases (FAISS, ChromaDB)
- Embeddings (Sentence-BERT)
- Retrieval metrics (Recall@k, MRR, NDCG)
- Chunk strategies
- Prompt engineering for RAG

**Pratik (P):**
- Document chunking & embedding
- Vector index creation
- Semantic search implementation
- LLM integration (Ollama)
- Evaluation pipeline
- Response quality assessment

**Ürün (X):**
- Production RAG system
- Document Q&A API
- Evaluation report
- Citation tracking

### Görevler

```bash
# 1. RAG pipeline
week4_rag/
├── chunker.py             # Document chunking
├── embedder.py            # Sentence-BERT embeddings
├── retriever.py           # Vector search
├── generator.py           # LLM integration (Ollama)
├── pipeline.py            # End-to-end RAG
└── evaluate.py            # RAG metrics

# 2. Data preparation
week4_rag/data/
├── documents/             # Source documents
│   ├── crypto_docs.md
│   └── ai_theory.md
└── eval/
    └── qa_pairs.json      # Evaluation Q&A

# 3. Infrastructure
week4_rag/infra/
├── vector_db.py           # FAISS wrapper
└── cache.py               # Embedding cache

# 4. Outputs
outputs/ai/week4/
├── vector_index.faiss
├── embeddings.pkl
├── rag_metrics.json
└── sample_responses.json
```

### RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────┐
│            RAG Pipeline                         │
├─────────────────────────────────────────────────┤
│                                                 │
│  User Query                                     │
│       ↓                                         │
│  ┌─────────────────┐                           │
│  │  Query Encoder  │ (Sentence-BERT)           │
│  │  query_vec      │                           │
│  └────────┬────────┘                           │
│           ↓                                     │
│  ┌─────────────────┐                           │
│  │  Vector Search  │ (FAISS)                   │
│  │  Top-k docs     │                           │
│  └────────┬────────┘                           │
│           ↓                                     │
│  ┌─────────────────┐                           │
│  │  Re-ranker      │ (optional)                │
│  │  Refine top-k   │                           │
│  └────────┬────────┘                           │
│           ↓                                     │
│  ┌─────────────────┐                           │
│  │  Prompt Builder │                           │
│  │  Context + Q    │                           │
│  └────────┬────────┘                           │
│           ↓                                     │
│  ┌─────────────────┐                           │
│  │  LLM Generator  │ (Ollama)                  │
│  │  Generate answer│                           │
│  └────────┬────────┘                           │
│           ↓                                     │
│  ┌─────────────────┐                           │
│  │  Citation       │                           │
│  │  Add sources    │                           │
│  └────────┬────────┘                           │
│           ↓                                     │
│       Response                                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### RAG Implementation Example

```python
# week4_rag/pipeline.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

class RAGPipeline:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def index_documents(self, documents):
        """Index documents for retrieval"""
        self.documents = documents
        embeddings = self.encoder.encode(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def retrieve(self, query, k=5):
        """Retrieve top-k relevant documents"""
        query_vec = self.encoder.encode([query])
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'document': self.documents[idx],
                'score': float(dist),
                'index': int(idx)
            })
        return results
    
    def generate(self, query, context):
        """Generate answer using Ollama"""
        prompt = f"""Context: {context}

Question: {query}

Answer based on the context above:"""
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama2',
                'prompt': prompt,
                'stream': False
            }
        )
        return response.json()['response']
    
    def answer(self, query, k=5):
        """End-to-end RAG pipeline"""
        # Retrieve
        results = self.retrieve(query, k=k)
        context = "\n\n".join([r['document'] for r in results])
        
        # Generate
        answer = self.generate(query, context)
        
        return {
            'answer': answer,
            'sources': results,
            'context': context
        }
```

### DoD (Definition of Done)

- [ ] Document chunking implemented (500-1000 tokens)
- [ ] Vector index created (1000+ chunks)
- [ ] Retrieval working (Recall@5 ≥ 60%)
- [ ] LLM integration (Ollama)
- [ ] End-to-end pipeline tested
- [ ] Evaluation on 50+ Q&A pairs
- [ ] Citation tracking functional
- [ ] Response quality ≥ 80% (human eval)
- [ ] API endpoint created (`/ask`)

### Çıktılar

```
reports/
└── ai/
    └── w4_rag.md

outputs/
└── ai/
    └── week4/
        ├── vector_index.faiss
        ├── embeddings.pkl
        ├── rag_metrics.json
        ├── sample_responses.json
        └── evaluation_results.csv
```

---

## Week 5 — 📘 Tool-calling Agent

**Hedef:** Multi-step reasoning agent, tool selection & execution, 2-step chain başarısı.

### Kazanımlar

**Teori (T):**
- Agent architectures (ReAct, MRKL)
- Tool calling patterns
- Function schemas (JSON Schema)
- Multi-step reasoning
- Error handling & retries
- Agent evaluation metrics

**Pratik (P):**
- Tool registry implementation
- LLM-based tool selection
- Execution engine
- Multi-step planning
- Conversation memory
- Agent testing

**Ürün (X):**
- Production agent framework
- 5+ tool implementations
- Agent API endpoint
- Execution traces

### Görevler

```bash
# 1. Agent framework
week5_agent/
├── agent.py               # Main agent class
├── tools/
│   ├── registry.py        # Tool registry
│   ├── calculator.py      # Math tool
│   ├── search.py          # Web search tool
│   ├── database.py        # DB query tool
│   └── crypto.py          # Crypto price tool
├── executor.py            # Tool execution engine
├── planner.py             # Multi-step planning
└── memory.py              # Conversation memory

# 2. Evaluation
week5_agent/eval/
├── test_cases.json        # Agent test cases
├── evaluate.py            # Agent evaluation
└── traces/                # Execution traces

# 3. Outputs
outputs/ai/week5/
├── agent_config.json
├── tool_schemas.json
├── execution_traces/
│   ├── trace_001.json
│   └── ...
└── metrics.json
```

### Agent Architecture

```python
# week5_agent/agent.py
from typing import List, Dict, Any
import json
import requests

class Tool:
    def __init__(self, name, description, parameters, func):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
    
    def to_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def execute(self, **kwargs):
        return self.func(**kwargs)

class Agent:
    def __init__(self, tools: List[Tool], llm_url="http://localhost:11434/api/generate"):
        self.tools = {tool.name: tool for tool in tools}
        self.llm_url = llm_url
        self.conversation_history = []
    
    def get_tool_schemas(self):
        return [tool.to_schema() for tool in self.tools.values()]
    
    def plan(self, query):
        """Ask LLM to plan tool usage"""
        tool_schemas = self.get_tool_schemas()
        prompt = f"""You are an AI assistant with access to tools.

Available tools:
{json.dumps(tool_schemas, indent=2)}

User query: {query}

Plan which tools to use and in what order. Respond in JSON format:
{{
  "plan": [
    {{"tool": "tool_name", "args": {{"arg1": "value1"}}, "reason": "why"}},
    ...
  ]
}}"""
        
        response = requests.post(
            self.llm_url,
            json={'model': 'llama2', 'prompt': prompt, 'stream': False}
        )
        
        plan_text = response.json()['response']
        # Extract JSON from response
        plan = json.loads(plan_text)
        return plan['plan']
    
    def execute_plan(self, plan):
        """Execute tool calls in sequence"""
        results = []
        for step in plan:
            tool_name = step['tool']
            args = step['args']
            
            if tool_name not in self.tools:
                results.append({'error': f'Unknown tool: {tool_name}'})
                continue
            
            try:
                result = self.tools[tool_name].execute(**args)
                results.append({'tool': tool_name, 'result': result})
            except Exception as e:
                results.append({'tool': tool_name, 'error': str(e)})
        
        return results
    
    def answer(self, query):
        """End-to-end agent execution"""
        # Plan
        plan = self.plan(query)
        
        # Execute
        results = self.execute_plan(plan)
        
        # Synthesize final answer
        final_prompt = f"""Query: {query}

Tool execution results:
{json.dumps(results, indent=2)}

Synthesize a final answer:"""
        
        response = requests.post(
            self.llm_url,
            json={'model': 'llama2', 'prompt': final_prompt, 'stream': False}
        )
        
        return {
            'answer': response.json()['response'],
            'plan': plan,
            'results': results
        }

# Example tools
def calculator(expression):
    """Evaluate math expression"""
    return eval(expression)  # Use safe_eval in production!

def get_crypto_price(symbol):
    """Get crypto price from CoinGecko"""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    response = requests.get(url)
    return response.json()

# Setup agent
tools = [
    Tool(
        name="calculator",
        description="Evaluate mathematical expressions",
        parameters={"expression": {"type": "string", "description": "Math expression"}},
        func=calculator
    ),
    Tool(
        name="get_crypto_price",
        description="Get current crypto price in USD",
        parameters={"symbol": {"type": "string", "description": "Crypto symbol (e.g., bitcoin)"}},
        func=get_crypto_price
    )
]

agent = Agent(tools=tools)
```

### DoD (Definition of Done)

- [ ] Agent framework implemented
- [ ] 5+ tools registered (calculator, search, DB, crypto, etc.)
- [ ] Tool selection working (LLM-based)
- [ ] Multi-step execution (2+ steps)
- [ ] Error handling & retries
- [ ] Conversation memory (5+ turns)
- [ ] Evaluation on 20+ test cases
- [ ] Success rate ≥ 70% (2-step chains)
- [ ] API endpoint `/agent/ask`

### Çıktılar

```
reports/
└── ai/
    └── w5_agent.md

outputs/
└── ai/
    └── week5/
        ├── agent_config.json
        ├── tool_schemas.json
        ├── execution_traces/
        │   ├── trace_001.json
        │   └── ...
        └── metrics.json
```

---

## Week 6 — 📗 LoRA Fine-tuning

**Hedef:** LoRA ile LLM fine-tuning, custom dataset, A/B test ≥ 60% win rate.

### Kazanımlar

**Teori (T):**
- Parameter-efficient fine-tuning (PEFT)
- LoRA (Low-Rank Adaptation) mechanics
- Rank selection & hyperparameters
- Target modules selection
- Evaluation strategies (A/B testing)
- Deployment considerations

**Pratik (P):**
- LoRA fine-tuning with HuggingFace PEFT
- Custom dataset preparation
- Training with limited resources
- Inference with adapters
- A/B testing setup
- Model merging

**Ürün (X):**
- Fine-tuned LLM (LoRA adapters)
- Custom instruction-following model
- A/B test results
- Deployment guide

### Görevler

```bash
# 1. LoRA training
week6_lora/
├── prepare_dataset.py     # Dataset preprocessing
├── train_lora.py          # LoRA fine-tuning
├── infer.py               # Inference with adapters
├── merge_adapters.py      # Merge LoRA weights
└── evaluate.py            # A/B testing

# 2. Dataset
week6_lora/data/
├── train.jsonl            # Training data (instruction-response)
├── val.jsonl              # Validation data
└── test.jsonl             # Test data

# 3. Outputs
outputs/ai/week6/
├── lora_adapters/
│   ├── adapter_config.json
│   └── adapter_model.bin
├── merged_model/
│   └── pytorch_model.bin
└── ab_test_results.json
```

### LoRA Training Script

```python
# week6_lora/train_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def main():
    # Load base model
    model_name = "meta-llama/Llama-2-7b-hf"  # or any open-source LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # Rank
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
        bias="none"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
    
    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": "week6_lora/data/train.jsonl",
        "validation": "week6_lora/data/val.jsonl"
    })
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="outputs/ai/week6/lora_adapters",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    # Train
    trainer.train()
    
    # Save adapters
    model.save_pretrained("outputs/ai/week6/lora_adapters")
    tokenizer.save_pretrained("outputs/ai/week6/lora_adapters")

if __name__ == "__main__":
    main()
```

### DoD (Definition of Done)

- [ ] LoRA adapters trained (rank 16, alpha 32)
- [ ] Custom dataset (500+ examples)
- [ ] Training completed (3 epochs)
- [ ] Inference working with adapters
- [ ] A/B test (base vs fine-tuned) on 50 prompts
- [ ] Win rate ≥ 60% (fine-tuned preferred)
- [ ] Adapter weights < 50MB
- [ ] Merged model created (optional)
- [ ] Deployment guide written

### Çıktılar

```
reports/
└── ai/
    └── w6_lora.md

outputs/
└── ai/
    └── week6/
        ├── lora_adapters/
        │   ├── adapter_config.json
        │   └── adapter_model.bin
        ├── merged_model/
        │   └── pytorch_model.bin
        ├── ab_test_results.json
        └── sample_outputs.txt
```

---

## Week 7 — 📘 Service & Monitoring

**Hedef:** Production FastAPI service, monitoring, CI/CD, p95 < 2.5s, error < 1%.

### Kazanımlar

**Teori (T):**
- FastAPI architecture
- Async request handling
- Model serving patterns
- Caching strategies
- Monitoring & observability
- CI/CD for ML models

**Pratik (P):**
- FastAPI service implementation
- Model loading & caching
- Prometheus metrics
- Grafana dashboards
- Load testing
- GitHub Actions CI/CD

**Ürün (X):**
- Production API
- Monitoring dashboard
- CI/CD pipeline
- Load test report

### Görevler

```bash
# 1. FastAPI service
week7_service/
├── app.py                 # Main FastAPI app
├── routers/
│   ├── predict.py         # Model inference endpoint
│   ├── rag.py             # RAG endpoint
│   └── agent.py           # Agent endpoint
├── models/
│   ├── loader.py          # Model loading utilities
│   └── cache.py           # Model cache
├── middleware/
│   ├── auth.py            # API key auth
│   └── metrics.py         # Prometheus metrics
└── config.py              # Configuration

# 2. Infrastructure
week7_service/infra/
├── docker/
│   ├── Dockerfile
│   └── compose.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboards/
│       └── api.json
└── k8s/                   # (optional)
    ├── deployment.yml
    └── service.yml

# 3. CI/CD
.github/workflows/
├── test_ai.yml            # Run tests
├── build_docker.yml       # Build & push Docker
└── deploy.yml             # Deploy to staging

# 4. Outputs
outputs/ai/week7/
├── load_test_report.html
├── grafana_screenshot.png
└── api_docs.pdf
```

### FastAPI Service Example

```python
# week7_service/app.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="NovaDev AI API", version="1.0.0")

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Model cache (singleton)
class ModelCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
        return cls._instance
    
    def load_model(self, model_name):
        if model_name not in self.models:
            # Load model (example)
            self.models[model_name] = torch.load(f"models/{model_name}.pt")
        return self.models[model_name]

model_cache = ModelCache()

# Request/Response models
class PredictRequest(BaseModel):
    model: str
    input_data: list

class PredictResponse(BaseModel):
    predictions: list
    latency_ms: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    import time
    start = time.time()
    
    try:
        model = model_cache.load_model(request.model)
        input_tensor = torch.tensor(request.input_data)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        predictions = output.tolist()
        latency_ms = (time.time() - start) * 1000
        
        return PredictResponse(predictions=predictions, latency_ms=latency_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    # Prometheus metrics endpoint
    pass
```

### DoD (Definition of Done)

- [ ] FastAPI service deployed
- [ ] 3+ endpoints (predict, RAG, agent)
- [ ] API key authentication
- [ ] Prometheus metrics exported
- [ ] Grafana dashboard created
- [ ] Load test: 100 req/s, p95 < 2.5s
- [ ] Error rate < 1%
- [ ] Docker image built & pushed
- [ ] CI/CD pipeline (test → build → deploy)
- [ ] API documentation (Swagger)

### Çıktılar

```
reports/
└── ai/
    └── w7_service.md

outputs/
└── ai/
    └── week7/
        ├── load_test_report.html
        ├── grafana_screenshot.png
        ├── api_docs.pdf
        └── performance_metrics.json
```

---

## Week 8 — 📗 Capstone E2E

**Hedef:** End-to-end integration, demo video (5 min), all systems working together.

### Kazanımlar

**Teori (T):**
- System integration patterns
- Deployment strategies
- Monitoring & alerting
- Incident response
- Documentation best practices

**Pratik (P):**
- Complete system integration
- Multi-service orchestration
- Load testing (production scale)
- Security hardening
- Documentation polish

**Ürün (X):**
- Complete AI + Crypto system
- 5-minute demo video
- Public documentation
- Release v1.0.0

### Görevler

```bash
# 1. Integration
week8_capstone/
├── orchestrator.py        # Coordinate all AI services
├── integration_tests.py   # E2E tests
└── scenarios/
    ├── scenario1.py       # Demo scenario 1
    ├── scenario2.py       # Demo scenario 2
    └── scenario3.py       # Demo scenario 3

# 2. Documentation
docs/
├── AI_SYSTEM_GUIDE.md     # Complete system guide
├── API_REFERENCE.md       # API documentation
├── DEPLOYMENT.md          # Deployment instructions
└── TROUBLESHOOTING.md     # Troubleshooting guide

# 3. Demo
public/
├── demo_video.mp4         # 5-minute demo
├── README.md              # Public-facing readme
└── screenshots/
    ├── dashboard.png
    ├── rag_query.png
    └── agent_execution.png

# 4. Outputs
outputs/ai/week8/
├── final_report.pdf
├── system_architecture.png
└── demo_scripts/
    ├── scenario1.md
    ├── scenario2.md
    └── scenario3.md
```

### Demo Scenarios

**Scenario 1: RAG Q&A**
```
User: "Explain early stopping in machine learning."
System:
  1. Retrieve relevant docs (Week 0 theory)
  2. Generate answer with citations
  3. Display sources
  4. Highlight: Fast retrieval (< 500ms), accurate citations
```

**Scenario 2: Agent Tool-calling**
```
User: "What's the price of Bitcoin and calculate 0.5 BTC in USD?"
System:
  1. Plan: [get_crypto_price, calculator]
  2. Execute: Fetch BTC price → Calculate
  3. Synthesize: "Bitcoin is $42,000. 0.5 BTC = $21,000"
  4. Highlight: Multi-step reasoning, tool integration
```

**Scenario 3: LoRA Fine-tuned Model**
```
User: "Generate a Turkish sentiment analysis for: 'Bu ürün harika!'"
System:
  1. Load fine-tuned Turkish BERT
  2. Classify: Positive (0.95 confidence)
  3. Explain: Top features
  4. Highlight: Custom fine-tuning, high accuracy
```

### DoD (Definition of Done)

- [ ] All AI services integrated
- [ ] 3 demo scenarios working
- [ ] 5-minute demo video recorded
- [ ] Public documentation complete
- [ ] API reference published
- [ ] Deployment guide written
- [ ] System architecture diagram
- [ ] Release v1.0.0 tagged
- [ ] CHANGELOG.md updated
- [ ] All 8 weeks documented

### Çıktılar

```
reports/
└── ai/
    └── w8_closeout.md

outputs/
└── ai/
    └── week8/
        ├── final_report.pdf
        ├── system_architecture.png
        └── demo_scripts/

public/
├── demo_video.mp4
├── README.md
└── screenshots/
```

---

## 🧠 Final Deliverables (Week 8 Sonu)

| Alan | Deliverable | Hedef | Status |
|------|-------------|-------|--------|
| **Linear Reg** | Val MSE ≤ 0.50 | ✅ Achieved | ⏳ |
| **MLP** | MNIST acc ≥ 97% | ✅ Achieved | ⏳ |
| **NLP** | Turkish F1 ≥ 0.85 | ✅ Achieved | ⏳ |
| **RAG** | Recall@5 ≥ 60% | ✅ Achieved | ⏳ |
| **Agent** | 2-step success 70% | ✅ Achieved | ⏳ |
| **LoRA** | A/B win rate ≥ 60% | ✅ Achieved | ⏳ |
| **Service** | API p95 < 2.5s | ✅ Achieved | ⏳ |
| **Capstone** | 5-min demo | ✅ Recorded | ⏳ |

---

## 🎓 Sertifikasyon — NovaBaron AI L1

### Kriterler

**Week 1-8 DoD:**
- [ ] All 8 weeks' DoD completed
- [ ] All models trained & deployed
- [ ] Production API functional
- [ ] Week 8 final report (15+ pages)
- [ ] 5-minute demo video uploaded

**Code Quality:**
- [ ] 100% test pass rate
- [ ] Ruff lint: 0 errors
- [ ] Code coverage ≥ 70%
- [ ] Documentation complete

**Public Presentation:**
- [ ] 20-minute final presentation
- [ ] Technical Q&A (10 min)
- [ ] Code walkthrough

### Sertifika Detayları

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║               NovaBaron AI Master — Level 1                ║
║                                                            ║
║  Sertifika Sahibi: [İsim]                                 ║
║  Tarih: [YYYY-MM-DD]                                       ║
║  Program: NovaDev v1.1 AI Track                           ║
║  Süre: 8 hafta (80-100 saat)                              ║
║                                                            ║
║  Yetenekler:                                               ║
║    ✓ ML fundamentals (scratch implementation)             ║
║    ✓ Deep Learning (MLP, CNN basics)                      ║
║    ✓ NLP (BERT fine-tuning)                               ║
║    ✓ RAG pipeline (retrieval + generation)                ║
║    ✓ Tool-calling agents (multi-step reasoning)           ║
║    ✓ LoRA fine-tuning (parameter-efficient)               ║
║    ✓ Production deployment (FastAPI + monitoring)         ║
║                                                            ║
║  Sertifika ID: NB-AI-2025-#0001                           ║
║  NFT: [Blockchain Address]                                ║
║                                                            ║
║  Baron's Signature: _______________________               ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 📚 Kaynaklar

### Books
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Pattern Recognition and Machine Learning" by Bishop
- "Speech and Language Processing" by Jurafsky & Martin

### Courses
- Stanford CS224N (NLP)
- Stanford CS229 (Machine Learning)
- Fast.ai Practical Deep Learning

### Papers
- "Attention Is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "LoRA: Low-Rank Adaptation of Large Language Models"
- "RAG: Retrieval-Augmented Generation"
- "ReAct: Synergizing Reasoning and Acting"

### Tools
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Weights & Biases](https://wandb.ai/)

---

## 🔗 İlgili Dosyalar

- [Week 0 Complete Documentation](../week0_setup/README.md)
- [Week 1 Master Plan](../WEEK1_MASTER_PLAN.md)
- [Crypto Track Outline](./CRYPTO_TRACK_OUTLINE.md)
- [Program Overview](./program_overview.md)

---

**Version:** 1.0  
**Last Updated:** 2025-10-06  
**Status:** Active (Week 0 Complete ✅, Week 1 Ready 👉)  
**Next:** Week 1 Linear Regression Sprint

---

🧠 **"From Zero to Production AI in 8 Weeks"** — NovaDev AI Mastery Track 🚀

