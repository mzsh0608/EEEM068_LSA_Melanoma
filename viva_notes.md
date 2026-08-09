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

TODO

---

## 5. ResNet18

TODO

---

## 6. ConvNeXt

TODO

---

## 7. BCEWithLogitsLoss

TODO

---

## 8. Weighted BCE

TODO

---

## 9. Evaluation metrics

TODO