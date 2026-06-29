# Master's Thesis — Task-Aware Speech Tokenization for Speech Translation

Controlled three-model comparison for speech-to-text translation (En->De, En->Zh)
on GigaST, with a frozen WavLM-Large encoder and a LoRA-tuned NLLB-200 decoder.

- **M1** continuous WavLM features (upper-bound baseline)
- **M2** offline task-agnostic quantization via K-means
- **M3** end-to-end task-aware Gumbel-Softmax PQ

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then put your HF_TOKEN in .env
```

## Run
```bash
python -m src.train --config configs/m1_continuous.yaml
```

## Structure
- `src/models/` encoder, bridges (M1/M2/M3), decoder
- `src/data/` GigaST join + collation
- `configs/` one YAML per experiment
- `notebooks/` exploration only
