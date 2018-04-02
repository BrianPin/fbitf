# fbitf
Fizzbuzz in tensorflow
This is inspired by http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
And adopting the pattern of mnist example in official tensorflow directory.

The result seems joyful:

```
Step 900 loss = 1.08 (0.001 sec)
Step 910 loss = 1.03 (0.001 sec)
Step 920 loss = 1.23 (0.001 sec)
Step 930 loss = 1.15 (0.001 sec)
Step 940 loss = 1.07 (0.000 sec)
Step 950 loss = 0.79 (0.001 sec)
Step 960 loss = 1.06 (0.001 sec)
Step 970 loss = 1.35 (0.001 sec)
Step 980 loss = 1.11 (0.001 sec)
Step 990 loss = 0.89 (0.001 sec)
Training Data Eval:
Num examples: 10000 Num correct: 5356 Precision @ 1: 0.5356
Validation Data Eval:
Num examples: 1000 Num correct: 523 Precision @ 1: 0.5230
Test Data Eval:
Num examples: 1000 Num correct: 528 Precision @ 1: 0.5280
```
