## Summary
This was the first open source version of DeepMind's [DQN paper](http://arxiv.org/abs/1312.5602). In addition, a crowd-based reward singal was collected which you can use to train your model, available here:

http://aiworld.io/data/space-invaders.html

## Details

All reinforcement learning done in Python. In addition, [solver.cpp](src/caffe/solver.cpp) was modified to support online observation of training data with `Solver<Dtype>::Solve` split into `OnlineUpdateSetup`, `OnlineUpdate`, and `OnlineForward` to set the input of the memory data layer, [determine the q-loss in `examples/dqn`](examples/dqn), then optionally backprop depending on whether we are training or just acting.

To use the crowd-reward data, download from above and set the following in your environment:

`export INTEGRATE_HUMAN_FEEDBACK=True`

Similar projects:
* https://github.com/mhauskn/dqn
* https://github.com/muupan/dqn-in-the-caffe

*Official improved DQN updated and released Feb 25th built on Torch*
* https://sites.google.com/a/deepmind.com/dqn/
