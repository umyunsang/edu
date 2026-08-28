# Wave 2: Hugging Face and W&B

## Hugging Face

Authenticated identity earlier resolved to `umyunsang`; public API verification on 2026-06-26 confirmed:

- Models: 7
  - `umyunsang/comfyui-models`
  - `umyunsang/civil-complaint-exaone-lora`
  - `umyunsang/civil-complaint-exaone-awq`
  - `umyunsang/GovOn-EXAONE-LoRA-v2`
  - `umyunsang/GovOn-EXAONE-Merged-v2`
  - `umyunsang/GovOn-EXAONE-AWQ-v2`
  - `umyunsang/govon-civil-adapter`
- Datasets: 2
  - `umyunsang/govon-civil-response-data`
  - `umyunsang/govon-legal-response-data`
- Spaces: 3
  - `umyunsang/govon-civil-adapter-train`
  - `umyunsang/govon-runtime`
  - `umyunsang/govon-multi-lora-test`

Public model-card checks support these safe claims:

- GovOn/EXAONE experiments include LoRA/QLoRA, merged, and AWQ surfaces.
- The civil adapter is for Korean government civil complaint draft responses and requires human review.
- The civil response dataset is public on Hugging Face and exposes train/validation splits.

Primary links:

- https://huggingface.co/umyunsang
- https://huggingface.co/umyunsang/govon-civil-adapter
- https://huggingface.co/datasets/umyunsang/govon-civil-response-data

## W&B

Public GraphQL verification on 2026-06-26 confirmed the `umyun3` entity and these public projects:

- `GovOn`
- `GovOn-retrain-v2`
- `civil-complaint-retrain-v2`
- `govon-evaluation`
- `govon-qlora-hparam-search`
- `civil-complaint-classification`
- `exaone-civil-complaint-public`
- `exaone-civil-complaint`
- `huggingface`

The `umyun3/GovOn` project has public run metadata, including finished and crashed runs. This supports saying that an experiment-tracking surface exists. It does not support claiming a performance result without deeper run inspection.

Primary link:

- https://wandb.ai/umyun3/GovOn

## LinkedIn Usage Boundary

Safe:

- "Hugging Face has GovOn/EXAONE model, dataset, and Space surfaces."
- "W&B has public GovOn experiment-tracking projects."
- "I am keeping model, dataset, and experiment-tracking artifacts visible."

Avoid:

- Unverified benchmark superiority.
- Any claim that all W&B runs succeeded.
- Any claim of production deployment from Hugging Face/W&B alone.
