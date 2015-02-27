Fork of [Caffe](http://caffe.berkeleyvision.org) created for replicating [DQN](http://arxiv.org/abs/1312.5602). 

All reinforcement learning done in Python. In addition, [solver.cpp](src/caffe/solver.cpp) was modified to support online observation of training data with `Solver<Dtype>::Solve` split into `OnlineUpdateSetup`, `OnlineUpdate`, and `OnlineForward` to set the input of the memory data layer, determine the q-loss in python, then optionally backprop depending on whether we are training or just acting.

Check in [examples/dqn](examples/dqn) for most of the relevant code.

Similar projects:
* https://github.com/mhauskn/dqn
* https://github.com/muupan/dqn-in-the-caffe

*Official improved DQN updated and released Feb 25th built on Torch*
* https://sites.google.com/a/deepmind.com/dqn/
