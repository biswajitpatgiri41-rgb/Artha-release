# ArthaAI LLM Strategy

## Selected release configuration
The shipped default is `rule-based-local-fallback`. It deterministically creates a bounded plan from the user question, is reproducible offline, and cannot invent economic facts. An optional OpenAI-compatible adapter is supported through explicit environment variables.

## Why no fine-tuning
Fine-tuning was not performed. The release benchmark targets planning, schema validity, tool selection, abstention, injection resistance, and grounding. A current data memorization strategy would be unsafe because economic observations change. If a future benchmark identifies a repeatable planning failure, use versioned supervised examples and LoRA/QLoRA, compare against this baseline on a held-out set, and retain rollback artifacts.

## Hardware
The build environment has CPU-only execution (6 vCPUs, approximately 18 GB RAM, no NVIDIA GPU). No local foundation model download is required. Hosted inference is optional and not needed for core forecasts.

## Provider abstraction
`app/arthaai/llm.py` exposes a common interface. Provider health, model identifier, timeouts, fallback behavior, and evaluation outputs are recorded. The application never treats provider output as numerical authority.
