# Table_App_02_HyperparameterSelection

| Parameter | B0 | M1 | M2 | Selection category | Systematically tuned? |
| --- | --- | --- | --- | --- | --- |
| image_size | 224 | 224 | 224 | predeclared_shared | No |
| batch_size | 32 | 32 | 32 | predeclared_shared | No |
| optimizer | adamw | adamw | adamw | predeclared_shared | No |
| learning_rate | 0.0001 | 0.0001 | 0.0001 | predeclared_shared | No |
| weight_decay | 0.0001 | 0.0001 | 0.0001 | predeclared_shared | No |
| loss | weighted_bce | weighted_bce | weighted_bce | predeclared_shared | No |
| pos_weight | 55.74304068522484 | 55.74304068522484 | 55.74304068522484 | data_derived | No |
| scheduler | none | none | none | predeclared_shared | No |
| max_epochs | 10 | 10 | 10 | predeclared_shared | No |
| early_stopping_patience | 3 | 3 | 3 | predeclared_shared | No |
| checkpoint_metric | roc_auc | roc_auc | roc_auc | predeclared_shared | No |
| primary_threshold | 0.5 | 0.5 | 0.5 | predeclared_shared | No |
| best_epoch | 4 | 4 | 8 | checkpoint_selected | No |
| phase_i_threshold_grid | not_applicable | [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] | [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] | posthoc_analysis_only | No |
| metadata_embedding_dim | not_applicable | not_applicable | 32 | architecture_specific | No |
| metadata_dropout | not_applicable | not_applicable | 0.2 | architecture_specific | No |

**Source note:** No systematic grid, random, or Bayesian search; common settings were deliberately matched.
