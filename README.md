# Robust Multi-modal Incomplete Data Completion Framework


##  Overview

We propose a robust method for multi-modal data completion and clustering, especially under high missing rates.

##  Structure

- `train.py`: Training script for low-missing rates（missing_rate=0.1）.
- `train1.py`: Enhanced training script for high-missing rates（missing_rate>=0.3）.
- `network`: Model architectures（missing_rate=0.1）.
- `network1`: Model architectures（missing_rate>=0.3）.
- `loss`: Loss functions（missing_rate=0.1）.
- `loss`: Loss functions（missing_rate>=0.3）.
- `data`: Dataset loading and processing.
- `metric`: Clustering evaluation metrics（missing_rate=0.1）.
- `metric1`: Clustering evaluation metrics（missing_rate>=0.3）.
- `dataloader`: load data（missing_rate=0.1）.
- `dataloader1`: load data（missing_rate>=0.3）.
##  Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
# or
python train1.py
```

