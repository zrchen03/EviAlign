# EviAlign: Learning Multimodal Embeddings with Evidence-Aligned Readout

EviAlign builds a single retrieval embedding from generated, task-relevant evidence. The model organizes evidence into five units—Entity, Attribute, Relation, Detail, and Summary—and reads the hidden state at each unit's closing token. The five states are averaged and normalized into one vector, so retrieval uses a standard single-vector index.

This repository provides a training-code example and an 800-pair sample illustrating the input and evidence-target format. The sample is not the full training set used for the paper's results.

## Repository contents

| Path | Contents |
| --- | --- |
| `qwenvl/train/` | Qwen-VL model loading, five-token readout, and joint training logic |
| `qwenvl/data/` | Paired query–candidate data loading and multimodal preprocessing |
| `scripts/train.sh` | Distributed training entry point and example hyperparameters |
| `scripts/zero3.json` | DeepSpeed ZeRO-3 configuration used by the training script |
| `data/evialign_sample.json` | 800 presentation-format query–candidate pairs from eight retrieval tasks |

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

Each record in `data/evialign_sample.json` has a `dataset_name`, a `qry` object, and a `pos` object. Both sides contain `conversations` with a `human` input and a `gpt` evidence target; image-bearing examples also have an `image` path relative to `data/`. The included `data/blank.jpg` is used by examples whose input side has no actual image.

The JSON file is a **presentation-format sample**, not a byte-for-byte export of the original training targets. Its field labels and line layout follow the paper. It does not include the MMEB image assets referenced by its relative paths.

## Setup

The provided launcher expects Linux, NVIDIA CUDA GPUs, `nvidia-smi`, and access to the Qwen3-VL-8B-Instruct model. Install the listed Python dependencies in a suitable CUDA environment:

```bash
pip install -r requirements.txt
```

Before training, place the corresponding MMEB images under `data/mmeb_v1/` so that the relative `image` paths in the JSON resolve. If using a different annotation file or image root, update the `EVIALIGN` entry in `qwenvl/data/__init__.py`. Check that all referenced image files exist before launching a distributed job.

## Training

For a single node, `scripts/train.sh` detects the local GPU count and launches one process per GPU:

```bash
bash scripts/train.sh
```

For multiple nodes, run the same command on each node with a shared master address and port, changing `NODE_RANK` for each node:

```bash
MASTER_ADDR=<master_ip> MASTER_PORT=8005 NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 bash scripts/train.sh
```

The example launcher selects `Qwen/Qwen3-VL-8B-Instruct`, BF16, DeepSpeed ZeRO-3, a per-device batch size of 4, gradient accumulation of 2, one epoch, and five evidence tokens. Checkpoints are written to `output/evialign/`. Adjust the model path, distributed settings, and training parameters in `scripts/train.sh` for your environment.

Running the launcher on the 800-pair sample checks the training pipeline, but does not reproduce the paper's results, which use approximately 500K query–candidate training pairs. The full evidence annotations, MMEB image assets, trained checkpoints, and evaluation scripts are not part of this example release.
