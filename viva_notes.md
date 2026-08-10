# EEEM068 LSA Viva Notes

## 1. Research problem

### Why is melanoma classification challenging?

- Severe class imbalance.
- False-negative melanoma predictions are important.
- Multiple images may originate from the same patient.
- Clinical metadata may provide information complementary to images.

---

## 2. Validation

### Why use patient-aware splitting?

Patient-aware splitting prevents observations belonging to the
same patient from appearing in both training and validation data.

This reduces the risk of information leakage and provides a more
realistic estimate of performance on unseen patients.

### What is patient leakage?

Patient leakage occurs when images belonging to the same
patient appear in both training and validation data.

A model may then exploit patient-specific similarities,
making validation performance appear better than its
performance on genuinely unseen patients.

### How was it prevented?

I used StratifiedGroupKFold.

- `target` was used for stratification.
- `patient_id` was used as the grouping variable.
- Five folds were generated.
- Fold 0 was used as the fixed validation set.
- The same split was used for every model.
- Known patient overlap between training and validation
  was explicitly verified to be zero.

### Did I perform full five-fold cross-validation?

No.

Five patient-aware folds were generated to define a
reproducible grouped partition, but only Fold 0 was used
as the held-out validation set for the LSA experiments.

This was done to maintain a common validation set across
experiments while keeping training computationally feasible.
---

## 3. Logistic Regression


### Logistic Regression

Logistic Regression is a linear probabilistic classifier.
It models the log-odds of the positive class as a weighted
linear combination of input features and maps the result
through the logistic sigmoid function.

In H0/H1, the input was a 4,096-dimensional vector created
by converting each lesion image to grayscale, resizing it
to 64x64, scaling pixel values to [0,1], and flattening it.

The main limitation is that flattening removes explicit
spatial structure. Logistic Regression does not learn
hierarchical image features in the way a CNN does.

### Class weighting

H1 used class_weight="balanced".

This gives minority-class observations greater influence
during model fitting. It changes the optimisation objective;
it is not the same as simply changing the classification
threshold afterward.

### Decision threshold

The predicted probability is converted into a class using
a threshold.

Lowering the threshold generally makes the classifier more
willing to predict melanoma, which can increase sensitivity
while also increasing false positives and reducing
specificity.

### ROC-AUC

ROC-AUC measures how well the model ranks positive cases
above negative cases across thresholds.

It is threshold-independent.

### Precision-recall / Average Precision

Precision-recall analysis is particularly useful for this
dataset because melanoma represents only about 1.76% of
observations.

### Why use the same Fold 0 for every model?

A common validation set means performance differences can
be attributed more meaningfully to model/method changes
rather than different validation samples.

---

## 4. CNNs

CNNs learn local image patterns with shared filters and build increasingly
abstract spatial features across layers. This preserves image structure,
unlike flattening grayscale pixels for Logistic Regression.

---

## 5. ResNet18

### Why ResNet18?

ResNet18 is a relatively compact, established CNN that supports ImageNet
transfer learning and is computationally practical for a controlled baseline.
It tests whether learned hierarchical RGB features improve over the flattened
grayscale H0/H1 representation without introducing a large architecture.

### What is a residual connection?

A residual block learns a transformation that is added to its input through a
skip connection. This gives gradients a direct route through the network and
helps deeper networks optimise without requiring every block to relearn an
identity mapping.

### Transfer learning and full fine-tuning

B0 started from `IMAGENET1K_V1` weights, replaced the 1,000-class classifier
with one binary logit, and updated the complete network. Pretraining supplies
general visual features; full fine-tuning adapts both those features and the
new classifier to lesion images.

---

## 6. ConvNeXt

ConvNeXt-Tiny is reserved for M1. It provides a meaningful controlled next
experiment because the fixed fold, resolution, preprocessing, weighted loss,
optimizer framework, and evaluation can remain unchanged while the principal
changed variable is the image architecture.

---

## 7. BCEWithLogitsLoss

### Why one logit and no sigmoid layer?

Binary classification needs one score: a positive logit favours melanoma and
a negative logit favours benign. `BCEWithLogitsLoss` combines sigmoid and
binary cross-entropy in a numerically stable calculation, so the model must
provide raw logits. Sigmoid is applied later when evaluation probabilities are
needed.

### Forward and backward passes

The forward pass maps a batch of images to logits and computes the weighted
loss against its labels. Backpropagation calculates the gradient of that loss
for every trainable parameter, after which AdamW updates the parameters.

---

## 8. Weighted BCE

### Why weight the positive class?

Only 467 of 26,499 training examples are melanoma. B0 used
`pos_weight = 26032 / 467 = 55.74304068522484`, so errors on positive training
examples contribute more strongly to the objective. This changes optimisation;
it is not threshold tuning or oversampling.

### Why is Fold 0 excluded from weighting?

`pos_weight` describes the training distribution and affects gradient updates.
Using Fold 0 labels in it would allow validation information to influence
fitting. The 6,627 Fold 0 samples were used only for epoch validation,
checkpoint selection, and final evaluation.

---

## 9. Evaluation metrics and B0 evidence

### Why AdamW and AMP?

AdamW provides adaptive parameter updates while applying decoupled weight
decay; B0 used learning rate and weight decay of 0.0001. On CUDA, AMP used
autocast and gradient scaling to reduce computation and memory demand while
protecting numerical stability.

### Why checkpoint by ROC-AUC?

ROC-AUC assesses ranking across thresholds and is less dominated by the large
benign majority than raw accuracy. The criterion was declared before training.
B0 therefore restored epoch 4, which had the highest validation ROC-AUC
(0.864276), even though Average Precision happened to peak at epoch 6.

### Why is accuracy misleading here?

H0 achieved 0.9810 accuracy while detecting only 1 of 117 melanomas. B0 had
lower accuracy (0.7681) but detected 93 melanomas, giving sensitivity 0.7949
and balanced accuracy 0.7812. Accuracy alone rewards majority-class
predictions in this imbalanced dataset.

### What happened during B0 training?

Training loss generally declined from 1.0615 to 0.8120. Validation ROC-AUC
peaked at epoch 4 and then fell over epochs 5-7 while validation loss rose.
Early stopping triggered after three non-improving epochs and restored epoch
4. This pattern is consistent with overfitting or instability, but the
fluctuations were not tested for statistical significance.

### What did B0 show compared with H0/H1?

B0 improved ROC-AUC to 0.8643 from 0.6570 (H0) and 0.6244 (H1), and Average
Precision to 0.1193 from 0.0386 and 0.0363. At threshold 0.5 it found 93
melanomas, versus 1 and 17, showing that the pretrained CNN did not collapse
to near-all-benign predictions. The cost was 1,513 false positives,
specificity of 0.7676, and precision of only 0.0579.

### What limitations remained?

B0 was evaluated on one fixed fold, without confidence intervals or external
validation. Weighted BCE improved sensitivity but left a substantial
sensitivity-specificity trade-off and low precision. No threshold was tuned,
so the reported operating point remains the common predeclared threshold 0.5.

---

## 10. Reusable PyTorch pipeline

### Dataset and DataLoader

A Dataset maps one index to one sample, including the image,
target, image name, and patient ID. A DataLoader wraps the
Dataset to create batches and control shuffling and worker-based
loading. Training is shuffled, while validation preserves a
deterministic order.

### Why is validation not augmented?

Validation uses only deterministic resizing and normalization.
Stochastic augmentation would change the evaluation samples
between passes and make model comparisons less reproducible.

### BCEWithLogitsLoss

The model produces an unbounded binary logit.
`BCEWithLogitsLoss` combines the sigmoid operation and binary
cross-entropy in a numerically stable implementation.

### Positive-class weighting

For weighted BCE, `pos_weight` is calculated from the training
partition as the number of negative examples divided by the
number of positive examples. This gives positive melanoma
examples greater contribution to the training loss without
using validation labels.

### Transfer learning and fine-tuning

B0 starts from ResNet18 weights pretrained on ImageNet and
replaces the 1,000-class classifier with one binary output.
The complete network will be fine-tuned rather than freezing the
pretrained backbone.

### Automatic mixed precision

On CUDA, autocast can use lower-precision operations where
appropriate, while gradient scaling reduces the risk of very
small gradients underflowing. This can reduce memory use and
computation time, but it does not guarantee identical results
across every platform.

### Why is sigmoid not part of the model?

The loss expects raw logits, so adding sigmoid inside the model
would duplicate that operation and reduce numerical stability.
Sigmoid is applied only when probabilities are needed during
validation or prediction.
