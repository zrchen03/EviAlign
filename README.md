# EviAlign: Learning Multimodal Embeddings with Evidence-Aligned Readout

EviAlign turns generated retrieval evidence into a single embedding. A shared multimodal model organizes evidence into five semantic units, reads the contextualized state at each unit's closing token, and averages and normalizes those states for standard single-vector retrieval.

`Multimodal input → five evidence units → boundary states → one retrieval vector`

This repository provides a reference training pipeline and representative evidence-annotation examples.

## Repository contents

| Path | Contents |
| --- | --- |
| [`qwenvl/train/`](qwenvl/train/) | Model loading, five-token readout, and joint training logic |
| [`qwenvl/data/`](qwenvl/data/) | Paired query–candidate data loading and multimodal preprocessing |
| [`scripts/train.sh`](scripts/train.sh) | Distributed training launcher |
| [`scripts/zero3.json`](scripts/zero3.json) | DeepSpeed ZeRO-3 configuration |
| [`data/evialign_sample.json`](data/evialign_sample.json) | Representative query–candidate evidence annotations |

## Evidence format

Each side of a training pair contains a multimodal input and an assistant target with five fields:

```text
[Entity] ... <ENT>
[Attribute] ... <ATT>
[Relation] ... <REL>
[Detail] ... <DET>
[Summary] ... <SUM>
```

The bracketed labels identify the evidence units for readers. The five angle-bracketed markers are special model tokens. The training code registers these tokens, supervises their generation along with the evidence text, and uses their final-layer hidden states for the retrieval embedding.

## Example annotations

`data/evialign_sample.json` contains representative query–candidate examples. Each record has a `dataset_name`, a `qry` object, and a `pos` object. Both sides contain `conversations` with a `human` input and a `gpt` evidence target; image-bearing examples also include an `image` path relative to `data/`. The included `data/blank.jpg` is used when an input side has no actual image.

The examples follow the paper's five-field presentation format. The reported experiments use approximately 500K query–candidate training pairs.

## Setup

The training launcher expects Linux, NVIDIA CUDA GPUs, `nvidia-smi`, and access to Qwen3-VL-8B-Instruct. Install the Python dependencies in a suitable CUDA environment:

```bash
pip install -r requirements.txt
```

Place the corresponding MMEB images under `data/mmeb_v1/` so that the relative `image` paths resolve. For a different annotation file or image root, update the `EVIALIGN` entry in `qwenvl/data/__init__.py`.

## Training

For a single node, `scripts/train.sh` detects the local GPU count and launches one process per GPU:

```bash
bash scripts/train.sh
```

For multiple nodes, run the same command on each node with a shared master address and port, changing `NODE_RANK` for each node:

```bash
MASTER_ADDR=<master_ip> MASTER_PORT=8005 NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 bash scripts/train.sh
```

The example launcher uses these settings:

| Setting | Value |
| --- | --- |
| Backbone | `Qwen/Qwen3-VL-8B-Instruct` |
| Precision | BF16 |
| Distributed training | DeepSpeed ZeRO-3 |
| Batch size per device | 4 |
| Gradient accumulation | 2 |
| Epochs | 1 |
| Evidence boundary tokens | 5 |

Checkpoints are written to `output/evialign/`. Adjust the model path, distributed settings, and training parameters in [`scripts/train.sh`](scripts/train.sh) for your environment.
