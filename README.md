# Knowledge Distillation Methods with Tensorflow
Knowledge distillation is the method to enhance student network by teacher knowledge.
So annually knowledge distillation methods have been proposed, but each paper's do experiments with different networks and compare with different methods.
Moreover, each method is implemented by each author, so if a new researcher wants to study knowledge distillation, they have to find or implement all of the methods. Surely it is tough work.
To reduce this burden, I publish some codes and modify from my research codes.
I'll update the code and knowledge distillation algorithm, and all of the things will be implemented using Tensorflow.



## Multi-connection Knowledge
Increases knowledge by sensing several points of the teacher network
- FitNet : To increase amounts of information, knowledge is defined by multi-connected networks and compared feature maps by L2-distance.
  - [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
