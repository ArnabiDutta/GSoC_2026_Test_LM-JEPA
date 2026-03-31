# SYMBA: LLM-JEPA for Feynman Squared Amplitudes [Project](https://ml4sci.org/gsoc/2026/proposal_SYMBA5.html)

**Contributor:** Arnabi Dutta

[JUPYTER NOTEBOOK](https://github.com/ArnabiDutta/GSoC_2026_Test_LM-JEPA/blob/main/ARNABI_DUTTA_GSoC_2026_Task.ipynb)

**Prerequisite Test:**
I have completed Tasks for Symbolic AI Projects GSoC 2026 Common Task 1.2, Specific Test 2.5.

This repository evaluates Joint Embedding Predictive Architectures (JEPA) and tokenization strategies for mapping high-energy physics amplitudes to their squared amplitudes (A → |A|^2). We experiment structural representations (Infix vs. Prefix notation) and evaluates the impact of physics-informed augmentations.


```text
## Repository Structure

.
├── ARNABI_DUTTA_GSoC_2026_Task.ipynb          # EDA, inference examples, and metric visualization
├── ARNABI_DUTTA_GSoC_2026_Task_full_code.ipynb # Consolidated codebase for quick review
├── check.py                                   # Data and model verification
├── data/
│   ├── dataset.py                             # Sequence formatting and batching
│   └── tokenizer.py                           # Lexer and Infix/Prefix Shunting-Yard tokenization
├── inference/
│   └── evaluate.py                            # Multi-tier evaluation (SymPy, Levenshtein, Exact Match)
├── models/
│   └── llm_jepa.py                            # Decoder-Only TransformerEncoder & dual-pass JEPA
├── scripts/
│   ├── 01_preprocess_data.py                  # Symmetry augmentation and string standardization
│   ├── 02_build_vocab.py                      # Shared vocabulary generation
│   └── 03_train_jepa.py                       # Main training executable
├── training/
│   └── trainer.py                             # Training loop, checkpointing, and W&B logging
└── train.sh                                   # End-to-end execution bash script
```

## Pre-Trained Model Weights

[Download Best Model Weights (.pt) Here](insert_link_here)

To evaluate locally, download to the `logs/` directory and run `inference/evaluate.py` using the `--wt_path` argument.


## Task 1.2: Data Pipeline & Tokenization

Provided dataset was limited for training a good transformer. To resolve this, I exploited Feynman diagram Permutation Symmetry. By extracting and locking kinematic variables (s_12, m_e, p_3), I applied seeded random shuffles to generic indices (`%gam_115` → `%gam_002`). I yielded a 15x dataset expansion without breaking physical topologies. A strict 80/10/10 split is enforced before augmentation to guarantee zero data leakage.

Standard BPE tokenizers shatter physics tensors into meaningless fragments. I implemented a regex lexer inspired by previous work to process entire kinematic tensors including subscripts and superscripts as single tokens:
`r'[a-zA-Z_][a-zA-Z0-9_]*(?:_{[^}]+})?(?:\([^)]+\))?(?:_[a-zA-Z0-9_]+)?(?:\^[a-zA-Z0-9_]+)?'`

Raw strings undergo a standardization pass before tokenization:
* `**` → `^`
* `^(*)` → `^CONJ` (prevents token shattering)
* `INT_NEG` → `NEG ` (forces operator behavior)
* `s_12` → `MANDELSTAM_12` (prevents alias collapse)

The tokens are then processed through a Shunting-Yard algorithm. This acts as a dynamic toggle, allowing the pipeline to output either standard Infix or tree-based Polish Prefix ASTs to compare learning efficiency.



## Test 2.5: Architecture & Training

The model is a Decoder-Only causal language model utilizing PyTorch's `nn.TransformerEncoder` with a strictly causal `torch.triu` mask, avoiding the cross-attention overhead of standard seq2seq setups.

Following the official LLM-JEPA implementation, the model executes two forward passes:
1. Generative pass using a standard causal mask where the target auto-regressively attends to the prompt.
2. JEPA pass using a 3D blocked mask where the target is strictly blinded from the source prompt.

The objective function minimizes:

$$\mathcal{L} = \text{CrossEntropy}(Y, \hat{Y}) + \lambda_{\text{jepa}} \left(1 - \text{CosineSimilarity}(h_{\text{text}}, h_{\text{code}})\right)$$

During validation, standard teacher-forcing accuracy naturally misaligns and grades structural boundary tokens (`[SEP]`, `[EOS]`). To fix this, shifted logits are dynamically sliced (`seq_preds = preds[:, t_size : t_size + c_size - 2]`) to strictly evaluate mathematical tokens.

The training script retains full dictionary states (`model_state_dict`, `optimizer`, `scheduler`, `epoch`) to seamlessly resume Cosine Annealing without learning rate decay resets.


## Inference & Multi-Tier Evaluation

Standard repetition penalties suppress valid operators that naturally recur in squared amplitudes. This was fixed by isolating the penalty strictly to `input_ids[prompt_len:]`, penalizing only the generated latent tree and not the prompt.

Because symbolic math can be written in multiple valid ways, token-level accuracy is insufficient. I evaluate generation using four metrics:
1. Exact Sequence Match
2. Token Accuracy (positional alignment)
3. Structural Match (Levenshtein edit distance)
4. Algebraic Match (`sympy.simplify(pred - gt) == 0`)

Evaluating Prefix models with SymPy requires a custom stack-based `prefix_to_infix` parser before ingestion. This solver is wrapped in a 2-second `signal.alarm` to kill infinite CPU hangs caused by invalid generated topologies.


## Quantitative Results

| Physics Model | Notation | Augmentation | Convergence (Epoch) | Token Accuracy | Structural Match (Levenshtein) | Sequence Accuracy (Exact) | Algebraic Match (SymPy) | Inference Speed (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **QED** | Prefix | 1x | 75 | 35.19% | 67.22% | 0.00% | **88.89%** | 1290.20 |
| **QED** | Prefix | 15x | 75 | 98.11% | 99.82% | 94.44% | **100.00%** | 1496.53 |
| **QED** | Infix | 1x | 211 | 81.94% | 96.51% | 63.89% | **63.89%** | 1151.13 |
| **QED** | Infix | 15x | 53 | 99.69% | 99.69% | 94.44% | **94.44%** | 1267.41 |
| **QCD** | Prefix | 1x | 108 | 9.45% | 39.74% | 0.00% | **50.00%** | 710.58 |
| **QCD** | Prefix | 15x | 137 | 87.13% | 93.58% | 79.17% | **79.17%** | 692.37 |
| **QCD** | Infix | 1x | 141 | 11.20% | 32.29% | 0.00% | **0.00%** | 686.49 |
| **QCD** | Infix | 15x | 6 | 5.48% | 21.93% | 0.00% | **0.00%** | 696.18 |

### Dataset & Vocabulary Statistics

The table below details the sample distributions and unique vocabulary sizes across all 8 experimental configurations. It explicitly validates that the $15\times$ permutation augmentation expands the training distribution.
| Physics Model | Notation | Augmentation | Base Samples | Train Size (Aug) | Val Size | Test Size | Unique Vocab Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **QED** | Infix | 1x (Baseline) | 360 | 288 | 36 | 36 | 279 |
| **QED** | Prefix | 1x (Baseline) | 360 | 288 | 36 | 36 | 282 |
| **QED** | Infix | 15x | 360 | 4,317 | 36 | 36 | 288 |
| **QED** | Prefix | 15x | 360 | 4,317 | 36 | 36 | 286 |
| **QCD** | Infix | 1x (Baseline) | 230 | 187 | 23 | 24 | 1,597 |
| **QCD** | Prefix | 1x (Baseline) | 230 | 187 | 23 | 24 | 1,575 |
| **QCD** | Infix | 15x | 230 | 2,805 | 23 | 24 | 2,584 |
| **QCD** | Prefix | 15x | 230 | 2,805 | 23 | 24 | 2,616 |
