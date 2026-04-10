# LLMOps Platform

A production-grade, self-hosted LLM serving platform with retrieval-augmented generation (RAG), configurable guardrails, prompt template versioning, automated evaluation, and full observability. Designed for teams that need to run large language models on their own infrastructure with enterprise controls around cost, quality, and safety.

---

## Architecture

```
                                    LLMOps Platform
    ============================================================================

    Documents (PDF, MD, HTML)
        |
        v
    +------------------+     +------------------+     +-------------------+
    | Document Loader  | --> |    Chunker       | --> |    Embedder       |
    | (ingestion)      |     | (recursive split)|     | (sentence-xfmr)  |
    +------------------+     +------------------+     +-------------------+
                                                            |
                                                            v
                                                    +----------------+
                                                    |    Qdrant      |
                                                    | (Vector Store) |
                                                    +-------+--------+
                                                            ^
                                                            |
    User Query                                              |
        |                                                   |
        v                                                   |
    +-----------+     +------------------+     +------------+----------+
    |  API      | --> | Input Guardrails | --> |   Retriever Service   |
    |  Gateway  |     | (injection,      |     |   (semantic search)   |
    |  (FastAPI)|     |  length, topics) |     +-----------------------+
    +-----------+     +------------------+               |
        |                                                v
        |                                    +----------------------+
        |                                    | Prompt Template      |
        |                                    | (Jinja2 + context)   |
        |                                    +----------+-----------+
        |                                               |
        |                                               v
        |                                    +----------------------+
        |                                    |   vLLM (GPU)         |
        |                                    |   Llama 3 8B         |
        |                                    |   Tensor Parallel    |
        |                                    +----------+-----------+
        |                                               |
        |                                               v
        |                                    +----------------------+
        |                                    | Output Filter        |
        |                                    | (toxic, profanity,   |
        |                                    |  hallucination)      |
        |                                    +----------+-----------+
        |                                               |
        |                                               v
        |                                    +----------------------+
        |                                    | PII Redaction        |
        |                                    | (email, phone, SSN,  |
        |                                    |  credit card, IP)    |
        |                                    +----------+-----------+
        |                                               |
        v                                               v
    +-------------------------------------------------------+
    |                    Response to User                    |
    +-------------------------------------------------------+

    Monitoring Layer (Prometheus + Grafana)
    ============================================================================
    | Latency (TTFT, generation) | Token Usage & Cost | Quality Metrics |
    | Error Rates (by type)      | GPU Utilization    | Guardrail Stats |
    ============================================================================
```

---

## Directory Structure

```
llmops-platform/
|-- infrastructure/
|   |-- terraform/
|   |   |-- gpu-nodepool/          # AKS/GKE GPU node pool provisioning
|   |   +-- storage/               # Persistent storage for models & vectors
|   +-- helm/
|       |-- vllm/                  # Helm chart: vLLM inference engine
|       |-- qdrant/                # Helm chart: Qdrant vector database
|       +-- llm-gateway/           # Helm chart: API gateway
|
|-- rag_pipeline/
|   |-- ingestion/
|   |   |-- document_loader.py     # Load documents from various formats
|   |   |-- chunker.py             # Recursive text chunking
|   |   |-- embedder.py            # Generate embeddings via sentence-transformers
|   |   +-- pipeline.yaml          # Ingestion pipeline configuration
|   +-- retriever/
|       |-- retriever_service.py   # FastAPI service for semantic search
|       +-- Dockerfile
|
|-- serving/
|   |-- vllm_config.yaml           # vLLM engine configuration
|   |-- deployment.yaml            # Kubernetes Deployment manifest
|   +-- hpa.yaml                   # Autoscaling on GPU util + queue depth
|
|-- guardrails/
|   |-- input_validator.py         # Prompt injection, length, topic blocking
|   |-- output_filter.py           # Toxic content, profanity, hallucination
|   +-- pii_redactor.py            # PII detection and redaction
|
|-- gateway/
|   |-- app.py                     # FastAPI gateway (main entry point)
|   |-- Dockerfile
|   +-- prompt_templates/
|       |-- customer_support.yaml  # Customer support agent template
|       +-- product_qa.yaml        # Product Q&A template
|
|-- evaluation/
|   |-- eval_dataset.jsonl         # Golden Q&A evaluation dataset
|   |-- run_eval.py                # Automated evaluation runner
|   +-- Jenkinsfile-eval           # CI pipeline for prompt template eval
|
|-- monitoring/
|   |-- dashboards/
|   |   |-- llm_latency.json       # TTFT, generation time, RPS, queue depth
|   |   |-- token_usage_cost.json  # Token counts, cost estimation, budgets
|   |   |-- quality_metrics.json   # Relevance, feedback, guardrail stats
|   |   +-- error_rates.json       # Errors by type, GPU mem, KV cache
|   +-- alerts/
|       |-- latency_spike.yaml     # TTFT, generation, queue depth alerts
|       |-- cost_budget_exceeded.yaml  # Daily cost and usage spike alerts
|       +-- quality_degradation.yaml   # Feedback, guardrails, retrieval alerts
|
|-- deployment/
|   |-- helm/
|   |   +-- llm-stack/             # Umbrella Helm chart (all components)
|   |       |-- Chart.yaml
|   |       +-- values.yaml
|   |-- canary/
|   |   +-- prompt-version-canary.yaml  # Istio canary routing for A/B testing
|   +-- Jenkinsfile                # Full CI/CD pipeline
|
|-- tests/
|   |-- conftest.py                # Shared fixtures (mocks, sample data)
|   |-- test_retriever.py          # Retriever service tests
|   |-- test_guardrails.py         # Input/output guardrail tests
|   |-- test_latency_sla.py        # Latency SLA compliance tests
|   +-- test_eval_regression.py    # Evaluation regression detection tests
|
|-- requirements.txt               # Python dependencies
+-- README.md                      # This file
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Application runtime |
| Docker | 24+ | Container builds |
| kubectl | 1.28+ | Kubernetes CLI |
| Helm | 3.13+ | Kubernetes package manager |
| GPU Nodes | NVIDIA A100/A10G | vLLM inference (2 GPUs per replica) |
| Qdrant | 1.8+ | Vector database for RAG |
| Istio | 1.20+ | Service mesh for canary routing (optional) |
| Prometheus | 2.48+ | Metrics collection |
| Grafana | 10+ | Dashboards and alerting |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ingest Documents

Load your knowledge base documents into the Qdrant vector store:

```bash
# Start Qdrant locally (or use your cluster)
docker run -p 6333:6333 qdrant/qdrant:v1.8.4

# Run the ingestion pipeline
python -m rag_pipeline.ingestion.embedder \
    --input-dir ./docs/ \
    --collection documents \
    --qdrant-url http://localhost:6333
```

### 3. Start the Retriever Service

```bash
export QDRANT_URL=http://localhost:6333
export COLLECTION_NAME=documents
export EMBEDDING_MODEL=all-MiniLM-L6-v2

uvicorn rag_pipeline.retriever.retriever_service:app \
    --host 0.0.0.0 --port 8081
```

### 4. Start the API Gateway

```bash
export VLLM_URL=http://localhost:8000       # Your vLLM instance
export RETRIEVER_URL=http://localhost:8081
export API_KEYS=my-dev-key-123

uvicorn gateway.app:app --host 0.0.0.0 --port 8080
```

### 5. Chat

```bash
curl -X POST http://localhost:8080/chat \
    -H "Content-Type: application/json" \
    -H "X-API-Key: my-dev-key-123" \
    -d '{
        "message": "What is the return policy for electronics?",
        "top_k": 5,
        "template": "customer_support"
    }'
```

---

## Deployment to Kubernetes

### Deploy the Full Stack with Helm

The umbrella chart deploys vLLM, Qdrant, and the gateway as a single release:

```bash
# Create the namespace
kubectl create namespace llmops

# Create required secrets
kubectl -n llmops create secret generic hf-token \
    --from-literal=token=hf_YOUR_TOKEN_HERE

kubectl -n llmops create secret generic llm-gateway-api-keys \
    --from-file=keys=api-keys.json

# Update Helm dependencies
helm dependency update deployment/helm/llm-stack

# Install the stack
helm install llm-stack deployment/helm/llm-stack \
    --namespace llmops \
    --values deployment/helm/llm-stack/values.yaml \
    --wait --timeout 10m
```

### GPU Node Pool (Terraform)

Provision GPU nodes for vLLM inference:

```bash
cd infrastructure/terraform/gpu-nodepool
terraform init
terraform plan -var="node_count=2" -var="vm_size=Standard_NC24ads_A100_v4"
terraform apply
```

### Verify Deployment

```bash
kubectl -n llmops get pods
kubectl -n llmops logs deploy/llm-stack-llm-gateway --tail=50
kubectl -n llmops port-forward svc/llm-stack-llm-gateway 8080:8080
```

---

## Guardrails Configuration

The platform includes three layers of guardrails:

### Input Validation (`guardrails/input_validator.py`)

- **Prompt injection detection**: Blocks attempts to override system instructions using regex-based pattern matching (e.g., "ignore previous instructions", "DAN mode", "jailbreak").
- **Maximum length enforcement**: Rejects inputs exceeding the configured character limit (default: 4096).
- **Forbidden topic blocking**: Blocks requests about dangerous topics (configurable list).
- **HTML sanitization**: Strips `<script>`, `<style>`, and other HTML tags before processing.

### Output Filtering (`guardrails/output_filter.py`)

- **Toxic content blocking**: Hard-blocks responses containing harmful content patterns.
- **Profanity redaction**: Replaces profanity with `[REDACTED]` tokens.
- **Response length limiting**: Truncates overly long responses.
- **Hallucination detection**: Flags responses with high ratios of hedging phrases.

### PII Redaction (`guardrails/pii_redactor.py`)

Detects and redacts PII before logging and optionally before returning responses:

| PII Type | Example | Redaction Token |
|----------|---------|-----------------|
| Email | `user@example.com` | `[EMAIL_REDACTED]` |
| Phone | `(555) 123-4567` | `[PHONE_REDACTED]` |
| SSN | `123-45-6789` | `[SSN_REDACTED]` |
| Credit Card | `4111-1111-1111-1111` | `[CREDIT_CARD_REDACTED]` |
| IP Address | `192.168.1.1` | `[IP_REDACTED]` |

---

## Prompt Template Versioning and A/B Testing

### Template Structure

Prompt templates are YAML files in `gateway/prompt_templates/`:

```yaml
version: "1.0"
name: "customer_support"
system_prompt: |
  You are a helpful customer support agent...
template: |
  Context: {{ context }}
  Question: {{ user_message }}
temperature: 0.3
max_tokens: 1024
```

### A/B Testing with Canary Routing

The platform supports A/B testing of prompt templates using Istio traffic splitting:

1. Deploy a canary gateway with the new prompt template version.
2. Apply the canary VirtualService:
   ```bash
   kubectl -n llmops apply -f deployment/canary/prompt-version-canary.yaml
   ```
3. Traffic is split 90/10 between stable (v1) and canary (v2).
4. Explicitly test the canary with a header:
   ```bash
   curl -H "x-prompt-version: v2" http://llm-gateway.company.com/chat ...
   ```
5. Monitor quality metrics in the Grafana quality dashboard.
6. Gradually increase canary weight or roll back.

---

## Evaluation Pipeline

### Running Evaluations

The evaluation suite tests LLM output quality against a golden Q&A dataset:

```bash
python evaluation/run_eval.py \
    --dataset evaluation/eval_dataset.jsonl \
    --gateway-url http://localhost:8080 \
    --api-key my-dev-key-123 \
    --baseline evaluation/baseline.json \
    --output evaluation/eval_report.json
```

### Metrics Computed

- **Answer relevance**: Cosine similarity between response and expected answer embeddings.
- **Faithfulness**: Ratio of source snippets referenced in the response.
- **Latency**: End-to-end response time (p50, p95, p99).

### Regression Detection

The eval runner compares current scores against a saved baseline. If any metric drops by more than 5%, the pipeline exits with a failure code, blocking the merge in CI.

### Saving a New Baseline

```bash
python evaluation/run_eval.py \
    --dataset evaluation/eval_dataset.jsonl \
    --gateway-url http://localhost:8080 \
    --output evaluation/eval_report.json \
    --save-baseline
```

---

## Cost Management

### Token-Level Cost Tracking

Every request records input and output token counts and computes an estimated cost:

- Default rates: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- Rates are configurable in the Grafana dashboard variables

### Budget Alerts

PrometheusRule alerts fire when:

- **Daily cost** exceeds the configured budget threshold (default: $100/day)
- **Token usage rate** spikes to >200% of the 7-day moving average for 1+ hour

### Rate Limiting

Per-API-key rate limiting is enforced at the gateway layer:

| Tier | Requests/min | Tokens/min |
|------|-------------|------------|
| Free | 10 | 10,000 |
| Standard | 60 | 100,000 |
| Enterprise | 600 | 1,000,000 |

---

## Monitoring and Alerting

### Grafana Dashboards

Import the JSON dashboards from `monitoring/dashboards/`:

| Dashboard | Key Panels |
|-----------|-----------|
| **LLM Latency** | TTFT p50/p95/p99, total generation time, RPS, queue depth, tokens/sec |
| **Token Usage & Cost** | Daily token counts, estimated cost, per-key usage, budget lines, cumulative spend |
| **Quality Metrics** | Eval relevance, user feedback ratio, guardrail block rate, response length, retrieval scores |
| **Error Rates** | Errors by type, rate limit hits per key, GPU memory, failed requests, KV cache utilization |

### PrometheusRule Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| `LLMTimeToFirstTokenHigh` | TTFT p95 > 500ms for 5min | Critical |
| `LLMGenerationSlow` | Generation p95 > 5s for 5min | Warning |
| `LLMQueueDepthHigh` | Queue > 50 for 3min | Warning |
| `LLMDailyCostExceeded` | Daily cost > $100 | Warning |
| `LLMTokenUsageSpike` | Usage > 200% of 7-day avg for 1hr | Warning |
| `LLMQualityDrop` | Feedback score < 0.6 for 24hr | Warning |
| `LLMHighGuardrailBlockRate` | Block rate > 15% for 1hr | Warning |
| `LLMRetrievalQualityLow` | Retrieval score < 0.5 for 6hr | Warning |

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# Retriever service tests
pytest tests/test_retriever.py -v

# Guardrail tests (input validation, output filtering, PII redaction)
pytest tests/test_guardrails.py -v

# Latency SLA compliance tests
pytest tests/test_latency_sla.py -v

# Evaluation regression tests
pytest tests/test_eval_regression.py -v
```

### CI/CD Pipeline

The full pipeline (`deployment/Jenkinsfile`) runs:

1. Build gateway and retriever Docker images
2. Push to container registry
3. Deploy the Helm stack to the cluster
4. Run the evaluation suite
5. Canary-route new prompt versions (if changed)
6. Monitor canary quality for 10 minutes
7. Full rollout or automatic rollback
8. Slack notification on success or failure

---

## License

Internal use only. See your organization's software licensing policy.
