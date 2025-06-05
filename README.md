---
library_name: transformers
license: apache-2.0
base_model: BSC-TeMU/roberta-base-bne
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: roberta-base-bne-platzi-project-nlp
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-bne-platzi-project-nlp

This model is a fine-tuned version of [BSC-TeMU/roberta-base-bne](https://huggingface.co/BSC-TeMU/roberta-base-bne) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4880
- Accuracy: 0.8508

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.356         | 1.0   | 2500 | 0.3989          | 0.8452   |
| 0.2499        | 2.0   | 5000 | 0.4880          | 0.8508   |


### Framework versions

- Transformers 4.48.3
- Pytorch 2.6.0+cu124
- Datasets 3.4.0
- Tokenizers 0.21.0
