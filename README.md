---

# GEGA

This repository contains codes for the paper “[GEGA: Graph Convolutional Networks and Evidence Retrieval Guided Attention for Enhanced Document-level Relation Extraction](https://arxiv.org/abs/2407.21384)”.

## Requirements

The following packages are required:

- Python (tested on 3.8.13)
- CUDA (tested on 11.6)
- PyTorch (tested on 1.11.0)
- Transformers (tested on 4.14.1)
- numpy (tested on 1.22.4)
- opt-einsum (tested on 3.3.0)
- wandb
- ujson
- tqdm

## Training

1. **Fully-supervised setting**  
   Run the following command for BERT training:  
   ```bash
   bash scripts/run_bert.sh ${name} ${lambda} 0.05
   ```

2. **Inference on distantly-supervised data**  
   Run the following command for BERT inference:  
   ```bash
   bash scripts/infer_distant_bert.sh ${name} ${load_dir}
   ```

3. **Self-Training on distantly-supervised data**  
   Run the following command for BERT self-training:  
   ```bash
   bash scripts/run_self_train_bert.sh ${name} ${teacher_signal_dir} ${lambda} ${seed}
   ```

4. **Fine-tuning on human-annotated data**  
   Run the following command for BERT fine-tuning:  
   ```bash
   bash scripts/run_finetune_bert.sh ${name} ${student_model_dir} ${lambda} ${seed}
   ```

## Evaluation

Run the following command for evaluation using BERT:  
```bash
bash scripts/isf_bert.sh ${name} ${model_dir} test
```

## Citation

If you make use of this code in your work, please kindly cite our paper:

```bibtex
@article{mao2024gega,
  title={GEGA: Graph Convolutional Networks and Evidence Retrieval Guided Attention for Enhanced Document-level Relation Extraction},
  author={Mao, Yanxu and Chen, Xiaohui and Liu, Peipei and Cui, Tiehan and Yue, Zuhui and Li, Zheng},
  journal={arXiv preprint arXiv:2407.21384},
  year={2024}
}
```
