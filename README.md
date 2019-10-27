
- FitNet paper 
  - [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
  
  
- To start the ditillation using the teacher saved in pre-trained/ResNet32.mat use --teacher=ResNet32 

- Example: guided layer is layer n# 3 and hint layer is n#11 , the bottelneck channel 	number here is 8. and max_pool_size of [4,4] window.
Is is important that guided layer and hint layer must have the same output size.

 !python train_w_distill.py --Distillation=FitNet --train_dir=fitnet --main_scope=Student_w_FitNet --teacher=ResNet_teacher  --hintLayerIndex=11 --guidedLayerIndex=3 --BottelneckChanelNBR=9 max_pool_btlnk_size=4 


- To retrain the teacher : put Distillation to None , and specify the main_scope=Teacher as follow :

!python train_w_distill.py --Distillation=None --train_dir=test --main_scope=Teacher

-To compute the trained student network performance, use:

!python computeValForStudentNet.py

 
