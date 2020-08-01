# GraphSGAN

- 在paper 原版改为了 tensorflow 版，数据预处理部分没变。一些api在找到了tf中对应的。
- linear在tf中没有参考 pytorch 文档实现了。
- train 的步骤改了下。

- 
训练了20 epoch, 也许哪里有问题, accuracy 后面没啥变化比较低。
'''
Iteration 17, loss_supervised = 1.9678, loss_unsupervised = 0.0016, loss_gen = 172.9809 train acc = 0.1461
Iteration 18, loss_supervised = 1.9367, loss_unsupervised = 0.0019, loss_gen = 169.1382 train acc = 0.1411
Iteration 19, loss_supervised = 1.9622, loss_unsupervised = 0.0017, loss_gen = 203.4629 train acc = 0.1419
'''
