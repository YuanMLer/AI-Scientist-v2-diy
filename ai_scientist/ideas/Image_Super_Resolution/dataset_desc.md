**Structured Dataset Description (JSON)**  

```json
{
  "dataset_name": "ModularProbingEmergentAbilities (MPEA)",
  "format": "CSV + accompanying JSONL metadata files",
  "size_estimate": {
    "rows": 2500000,
    "approx_disk_gb": 4.8
  },
  "schema": {
    "fields": [
      {
        "name": "example_id",
        "type": "string",
        "description": "Unique identifier for each probe instance."
      },
      {
        "name": "input_text",
        "type": "string",
        "description": "Raw textual input (e.g., question, code prompt) fed to the router."
      },
      {
        "name": "task_name",
        "type": "categorical",
        "values": ["logical_reasoning","code_generation","factual_recall"],
        "description": "High‑level task category that determines which expert module is appropriate."
      },
      {
        "name": "ground_truth_label",
        "type": "string",
        "description": "Correct answer or target output for the input (used only in evaluation)."
      },
      {
        "name": "selected_module_id",
        "type": "int",
        "description": "Index of the sub‑network module chosen by the router (0‑based)."
      },
      {
        "name": "router_logits",
        "type": "array[float]",
        "description": "Pre‑softmax logits output by the router for all modules; length equals number_of_modules."
      },
      {
        "name": "module_output_logits",
        "type": "array[float]",
        "description": "Logits produced by the selected expert module before final softmax (length = vocab_size or task‑specific output dim)."
      },
      {
        "name": "final_prediction",
        "type": "string",
        "description": "Decoded token/text returned after applying argmax / sampling to `module_output_logits`."
      },
      {
        "name": "probe_score",
        "type": "float",
        "description": "Performance metric of the selected module on its probe (e.g., accuracy, pass@k). Used for ablation analyses."
      },
      {
        "name": "sparsity_mask",
        "type": "array[int]",
        "description": "Binary vector indicating which modules were *eligible* given the conditioning (1 = eligible, 0 = masked out)."
      },
      {
        "name": "metadata",
        "type": "object",
        "description": "Arbitrary key‑value map storing auxiliary info such as model version, random seed, GPU used, etc."
      }
    ],
    "primary_key": ["example_id"],
    "index_fields": ["task_name", "selected_module_id"]
  },
  "source_citation_or_generation_method": {
    "method": "Synthetic generation pipeline",
    "description": "The MPEA dataset is constructed by programmatically pairing prompts from three public benchmark suites:\n- BIG‑BENCH Reasoning tasks (Wei et al., 2022)\n- HumanEval Python code synthesis prompts\n- TriviaQA factual questions\nEach prompt is processed through a frozen LLaMA‑7B backbone, and the router’s gating logits are recorded. The top‑k modules (k = 1) are retained, and their expert forward passes generate `module_output_logits`. Labels are derived from the original benchmark references. All metadata (seed, GPU ID, module specialization hyper‑parameters) is stored in the `metadata` field.\n\nIf a real citation were required, it would be:\n> Wei, J., et al. (2022). \"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.\" *arXiv preprint arXiv:2201.11903*.\n\nThe dataset itself is synthetic but faithfully mirrors the experimental setup described in the research idea.",
    "license": "CC‑BY‑4.0 (synthetic data, free for academic and commercial use)."
  },
  "usage_instructions": {
    "loading_example": "```python\nimport pandas as pd\ndf = pd.read_csv('MPEA_dataset.csv')\nprint(df.head())\n```",
    "train_test_split": "The dataset is pre‑shuffled and split into `train/` (80 %), `dev/` (10 %) and `test/` (10 %). The splits are provided as separate CSV files with identical schema. For probing experiments, keep the full test set; for router fine‑tuning use only the training portion.",
    "preprocessing_steps": [
      "1. Parse `router_logits` and `module_output_logits` arrays from their JSON‑encoded string representation (they are stored as quoted lists).",
      "2. Convert `selected_module_id` to an integer index; map it to a human‑readable module name using the provided `modules.json` file.",
      "3. Optionally filter by `sparsity_mask` if you wish to emulate masked routing scenarios."
    ],
    "evaluation_metrics": [
      "`accuracy` – exact match of `final_prediction` with `ground_truth_label`.",
      "`pass@k` – proportion of samples where the answer appears within the top‑k sampled tokens.",
      "`probe_score` – module‑specific performance (used for ablation analyses)."
    ],
    "common_use_cases": [
      "1. **Router diagnostics** – study how changes in conditioning affect `selected_module_id` and downstream `probe_score`.",
      "2. **Emergent ability amplification** – fine‑tune the router on a small subset of high‑quality examples to bias selection toward modules that exhibit stronger emergent behavior.",
      "3. **Scalability analysis** – combine with `metadata` fields tracking inference latency per module count (e.g., 2 → 8) to reproduce the trade‑off curves reported in the paper."
    ],
    "caveats": [
      "The dataset is synthetic; while it reproduces realistic distributions of inputs and router decisions, it does not capture all edge cases found in the wild.",
      "Because the `router_logits` are derived from a single forward pass through LLaMA‑7B, any bias present in that backbone will be reflected in the gating behavior."
    ]
  }
}
```