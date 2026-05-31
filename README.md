# CIgFlow — Conditional Immunoglobulin Flow Matching

CIgFlow is a deep-learning framework for **antigen-conditioned antibody sequence generation**. It combines a **Conditional Flow Matching (CFM)** model with a **GPT-2-style autoregressive decoder** to generate novel, antibody heavy-chain sequences targeting a specific antigen. The approach learns a continuous mapping from a joint antibody–antigen embedding space to realistic antibody sequences.

---

## How It Works

```
Antibody + Antigen sequence
         │
         ▼
   IgGen (p-IgGen) Encoder
         │  produces embeddings
         ▼
  Concatenated embedding
  [ab_emb ‖ ag_emb]
         │
         ▼
  Conditional Flow Matching Model
  (learns to transport noise → target latent)
         │  Heun ODE solver
         ▼
     Latent vector z
         │
         ▼
  GPT-2-like Autoregressive Decoder
         │  top-k / nucleus sampling
         ▼
  Generated antibody sequence (FASTA)
```

The two-stage pipeline separates **latent space navigation** (flow model) from **sequence decoding** (transformer decoder), allowing fine-grained control over diversity, antigen specificity, and generation quality.

---

## Repository Structure

```
CIgFlow/
├── scripts/
│   ├── main.py                     # Training entry point
│   ├── nanobody_generator.py       # Core generator class (embedding, training, generation)
│   ├── conditional_flow_matching.py# CFM model + Heun sampler
│   ├── gpt2_decoder.py             # GPT-2-style autoregressive decoder
│   ├── transformer_block.py        # Transformer building block
│   ├── positional_encoding.py      # Sinusoidal positional encoding
│   └── trainers.py                 # Training loops for flow model and decoder
├── Inference/
│   └── inference.py                # Load saved model and run generation
└── Generated Sequence/
    ├── 1000_generated_Antibodies_HER2.fasta
    ├── 1000_generated_Antibodies_SARs-Cov2.fasta
    ├── 1000_ablation_no_flow_HER2.fasta
    ├── 1000_ablation_no_flow_SARs-COV2.fasta
    ├── 1000_generated_zero_guided_Antibodies_SARs-Cov2.fasta
    ├── 1000_generated_IHLM_SARs_humaness.csv
    ├── *_humanness.csv             # OASis humanness scores
    └── *_solubility.txt            # Solubility evaluation results
```

---

## Key Components

### `NanobodyGenerator` (`nanobody_generator.py`)
The central orchestrator. Responsibilities:
- Loads the **p-IgGen** protein language model (based on `ollieturnbull/p-IgGen`) to compute sequence embeddings.
- Encodes antibody–antigen pairs into a joint embedding via concatenation.
- Initialises, trains, and runs both the flow model and the decoder.
- Provides `generate_sequence()` and `generate_multiple_sequences()` with temperature, top-k, top-p, and classifier-free guidance controls.

### `ConditionalFlowMatchingModel` (`conditional_flow_matching.py`)
A residual MLP that learns the **conditional vector field** between noise and the target embedding distribution. Key features:
- Sinusoidal time embedding for continuous-time conditioning.
- Classifier-free guidance (CFG) with a configurable dropout rate for the antigen context.
- Standard CFM loss and a variance-preserving (VP-CFM) loss variant.
- **Heun's method** ODE sampler for high-quality deterministic generation.

### `GPT2LikeDecoder` (`gpt2_decoder.py`)
A causal transformer decoder that maps a latent vector to an antibody amino-acid sequence token-by-token. Uses:
- Latent projection layer to initialise generation from the flow model's output.
- Multi-head self-attention with causal masking.
- Top-k and nucleus (top-p) sampling at each step.

### `trainers.py`
- `FlowMatchingTrainer` — wraps the flow model optimizer with an Exponential Moving Average (EMA) of weights.
- `GPT2DecoderTrainer` — cross-entropy training loop with perplexity tracking.

---

## Installation

```bash
git clone https://github.com/FairuzShadmaniShishir/CIgFlow.git
cd CIgFlow

# Recommended: create a conda or virtualenv environment
pip install torch transformers scikit-learn pandas tqdm
```

> The project was developed with a dedicated `Masters_Thesis` conda environment (see comment at the top of `main.py`).

---

## Training

Edit the data paths in `scripts/main.py` to point to your `.parquet` antibody–antigen dataset (e.g. the ASD — Antigen Specific Antibody Database), then run:

```bash
cd scripts
python main.py
```

**Expected dataset columns:**
| Column | Description |
|---|---|
| `dataset` | Dataset name (e.g. `covid-19`, `buzz`) |
| `heavy_sequence` | Antibody heavy-chain amino-acid sequence |
| `antigen_sequence` | Antigen amino-acid sequence |

Training proceeds in two stages:
1. **Flow model** — trained for 50 epochs (configurable) on the joint embedding space.
2. **Decoder** — trained for 50 epochs with perplexity monitoring.

---

## Inference

Load a saved model and generate sequences against a new antigen:

```python
from Inference.inference import run_inference

antibody_seed = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
antigen       = "MELAALCRWGLLLALLPPGAASTQ..."  # HER2 ECD

sequences = run_inference(antibody_seed, antigen)

# Save to FASTA
with open("generated.fasta", "w") as f:
    for i, seq in enumerate(sequences):
        f.write(f">Sequence_{i+1}\n{seq}\n")
```

Generation parameters (all configurable):

| Parameter | Default | Description |
|---|---|---|
| `num_sequences` | 1000 | Number of sequences to generate |
| `temperature` | 0.7 | Sampling temperature |
| `num_steps` | 100 | ODE integration steps (Heun) |
| `noise_scale` | 1.0 | Initial noise magnitude |
| `guidance_scale` | 0.0 | Classifier-free guidance strength |
| `top_k` | 50 | Top-k token filtering |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `max_len` | 256 | Maximum sequence length |

---

## Pre-generated Sequences

The `Generated Sequence/` folder contains 1 000-sequence FASTA files for two benchmark targets:

| File | Target | Notes |
|---|---|---|
| `1000_generated_Antibodies_HER2.fasta` | HER2 | Main CIgFlow output |
| `1000_generated_Antibodies_SARs-Cov2.fasta` | SARS-CoV-2 Spike | Main CIgFlow output |
| `1000_ablation_no_flow_HER2.fasta` | HER2 | Ablation — decoder only, no flow model |
| `1000_ablation_no_flow_SARs-COV2.fasta` | SARS-CoV-2 | Ablation — decoder only |
| `1000_generated_zero_guided_Antibodies_SARs-Cov2.fasta` | SARS-CoV-2 | Zero guidance (guidance_scale=0) |

Accompanying evaluation files (OASis humanness scores, solubility predictions) are included for each set.

---

## Evaluation

Generated sequences can be evaluated with **PROMB OASIS** for humanness scoring:

```bash
promb oasis -o scores.csv generated.fasta
```

The `main.py` script includes helper functions (`run_promb_evaluation`, `analyze_scores`, `filter_top_sequences`) to automate this workflow and surface the top-ranked candidates.

---

## Citation / Acknowledgements

- Embedding model: [p-IgGen](https://huggingface.co/ollieturnbull/p-IgGen)
- Training data: ASD — Antigen Specific Antibody Database
- Humanness evaluation: [PROMB OASis](https://promb.naturalantibody.com/)
- Flow matching methodology: Lipman et al., *Flow Matching for Generative Modeling* (2022)
