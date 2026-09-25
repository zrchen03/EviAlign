# Learning Multimodal Embeddings With Evidence-Aligned Readout

## Installation

```bash
pip install -r requirements.txt
```

## Training

Single-node:

```bash
bash scripts/train.sh
```

Multi-node (run on every node, change only `NODE_RANK`):

```bash
MASTER_ADDR=<master_ip> MASTER_PORT=8005 NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 bash scripts/train.sh
```

Checkpoints are saved to `./output/evialign/`.

`data/evialign_sample.json` contains presentation-format examples of the five evidence fields. Its field labels and line layout follow the paper and are not a byte-for-byte export of the original training targets.
