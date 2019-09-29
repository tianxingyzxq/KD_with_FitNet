
- FitNet : paper 
  - [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
  
  
To start the ditillation using the teacher saved in pre-trained/ResNet32.mat use --teacher=ResNet32 

!python train_w_distill.py --Distillation=FitNet --train_dir=fitnet --main_scope=Student_w_FitNet --teacher=ResNet32 

To retrain the teacher : put Distillation to None , and specify the main_scope=Teacher as follow :

!python train_w_distill.py --Distillation=None --train_dir=test --main_scope=Teacher

To compute the trained student network performance, use:

!python computeValForStudentNet.py

 
