# TransferLearning

Implemented transferlearning for an EfficientNetV2L on the tf.flower dataset. The EfficientNetV2L was pretrained on the Imagenet dataset. 

## Model without Augmentation

Accuracy on test set is:0.9482288956642151
Loss on test set is:0.18121804296970367

Confusion matrix, without normalization
[[121   0   0   1   0]
 [  3 154   0   4   0]
 [  1   1 118   0   8]
 [  1   2   2 144   1]
 [  2   1   9   2 159]]


Classification report:
               precision    recall  f1-score   support
           0       0.95      0.99      0.97       122
           1       0.97      0.96      0.97       161
           2       0.91      0.92      0.92       128
           3       0.95      0.96      0.96       150
           4       0.95      0.92      0.93       173

   accuracy                            0.95       734
   macro avg       0.95      0.95      0.95       734
weighted avg       0.95      0.95      0.95       734


## Model without Augmentation

Accuracy on test set is:0.9509536623954773
Loss on test set is:0.17963309586048126

Confusion matrix, without normalization
[[120   1   0   0   1]
 [  3 154   1   3   0]
 [  1   0 118   0   9]
 [  1   4   0 144   1]
 [  2   1   7   1 162]]

Classification report:
               precision    recall  f1-score   support
           0       0.94      0.98      0.96       122
           1       0.96      0.96      0.96       161
           2       0.94      0.92      0.93       128
           3       0.97      0.96      0.97       150
           4       0.94      0.94      0.94       173

   accuracy                            0.95       734
   macro avg       0.95      0.95      0.95       734
weighted avg       0.95      0.95      0.95       734

