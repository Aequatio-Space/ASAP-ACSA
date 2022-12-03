# Multi-task Sentimental Analysis based on Convolutional Neural Networks
## Basic Information
Final Project for course “Knowledge Engineering", Spring semester 2022, Beijing Institute of Technology.
## codebase details
+ `preprocess.py`: contains the `Vocab` class for recording word-frequency table, also includes functions for reading files, parsing strings, etc.
+ `model.py`: contains all model used in experiments. ACSA for aspect catogory sentimental analysis, and RP for rating prediction.
  + GCAEReg(ACSA): A GCAE with Dual Conv layer, and output a regressed value.
  + GCAEReg-MultiConv(ACSA): a GCAE with multiple conv layers and output a regressed value.
  + GCAE-ACSA-RL(-CLS)(RP): a GCAE that has transfer learning from ACSA, with both regression and classification version included.
  + GCAE-Lite(RP): a reproduction of TextCNN.
  + BiRNN(RP): a bidirectional LSTM network with Pytorch libraries.
  + GCAE-ACSA: GCAE+, the network capable of completing both tasks.
+ `train_utils.py`: contains label-smoothing function, different evaluation functions and collcate function for dataset processing.
+ `trainFunc.py`: includes main train loop for different tasks, which include logging and checkpoint saving.
+ `plot.py`: the main function for calling `collector` class for figures.
+ `eval_formal.py`: call evaluation functions for model performance.
+ `ACSA+RL+Join+renewed.py`: the main script for running transfer learning on GCAE+.
+ `textCNN-std.py`: an attempt to reproduce TextCNN on RP task.
+ `textCNN-label-smoothing.py`: the best performance of rating prediction task is completed with this script.
+ `meituan-asap-aspect-renewed.ipynb`: trained with Kaggle GPU, the best performance of ACSA task is completed with this script.
+ `textCNN-Reg.py`: The script used to study classification and regression tasks.
