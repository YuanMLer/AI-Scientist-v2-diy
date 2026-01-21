# Configuration Migration Report

## Overview
All Large Language Model (LLM) and Vision Language Model (VLM) configurations have been migrated from scattered hardcoded values and local config files to a centralized configuration file: `bfts_config.yaml`.

This migration ensures:
- No hardcoded API keys or endpoints in source code.
- Dynamic configuration loading.
- Centralized management of model providers (OpenAI, Anthropic, Ollama, etc.).
- Environment variable integration for security.

## Migration Details

### 1. Central Configuration File
**File:** `bfts_config.yaml` (Project Root)
- **New Section:** `llm_config`
- **Features:**
    - Defines `models` map with specific settings for each model.
    - Defines `defaults` for dynamic model names (e.g., `ollama/*`).
    - Uses OmegaConf interpolation `${oc.env:VAR_NAME}` to securely load API keys from `.env`.

### 2. Codebase Changes

#### A. LLM Client Creation (`ai_scientist/llm.py`)
- **Original:** `create_client` function had hardcoded logic for 'claude', 'openai', 'deepseek', etc., reading directly from `os.environ`.
- **New:** Uses `ai_scientist.config_loader.get_llm_config(model)` to retrieve configuration.
- **Benefit:** Adding a new model requires only YAML changes, no code edits.

#### B. VLM Client Creation (`ai_scientist/vlm.py`)
- **Original:** Hardcoded `AVAILABLE_VLMS` list and `create_client` function.
- **New:** `AVAILABLE_VLMS` dynamically populated from config. `create_client` uses `get_llm_config`.

#### C. Tree Search Backends
- **OpenAI Backend (`ai_scientist/treesearch/backend/backend_openai.py`):**
    - **Original:** `get_ai_client` used arguments or environment variables directly.
    - **New:** Fetches API key and base URL from `bfts_config.yaml`.
- **Anthropic Backend (`ai_scientist/treesearch/backend/backend_anthropic.py`):**
    - **Original:** Hardcoded to use `AnthropicBedrock`.
    - **New:** Checks `provider` in config. Supports both `anthropic` (standard) and `bedrock`.

#### D. Tree Search Configuration (`ai_scientist/treesearch/utils/config.py`)
- **Original:** Loaded a local `config.yaml` (which was missing or separate).
- **New:** Updated to load the centralized `bfts_config.yaml` via `ai_scientist.config_loader`.

### 3. New Configuration Usage

To add a new model, update `bfts_config.yaml`:

```yaml
llm_config:
  models:
    "new-model-name":
      provider: "openai" # or ollama, anthropic
      api_key: "${oc.env:NEW_MODEL_KEY}"
      base_url: "https://api.example.com/v1"
```

For Ollama models, any model starting with `ollama/` will automatically use the default configuration defined in `llm_config.defaults.ollama` unless explicitly overridden.

### 4. Environment Variables
API keys should be placed in `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434/v1
```

These are automatically loaded by the application.
