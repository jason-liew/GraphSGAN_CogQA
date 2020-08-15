终于折腾好了在服务器上跑起来baseline.
完整命令:
nvidia-docker run --net=host --volume="$HOME/workspace:/tf" --user 0:0 -it chenqibin422/minerl bash

但是完整跑完时间较久，所以只跑了几个episode.

baseline1: ppo

```
INFO     - 2020-08-15 06:01:56,887 - [chainerrl.experiments.train_agent train_agent 59] outdir:results/MineRLTreechop-v0/ppo/20200815T054120.495466 step:4000 episode:1 R:8.0
INFO     - 2020-08-15 06:01:56,888 - [chainerrl.experiments.train_agent train_agent 60] statistics:[('average_value', 0.025590062258881517), ('average_entropy', 1.538290573477745), ('average_value_loss', 0.05641872310661711), ('average_policy_loss', -0.07323072407598374), ('n_updates', 96), ('explained_variance', 0.13813830775879976)]
```
大概说
ppo 是在PG（policy gradient）算法基础上做了改进，提出了新的目标函数可以在多个训练步骤实现小批量的更新，可以更好的利用现有的样本更好的收敛。

baseline2: rainbow
```
INFO     - 2020-08-15 10:31:34,830 - [chainerrl.experiments.train_agent train_agent 59] outdir:results/MineRLTreechop-v0/rainbow/20200815T101210.692206 step:4000 episode:1 R:12.0
INFO     - 2020-08-15 10:31:34,830 - [chainerrl.experiments.train_agent train_agent 60] statistics:[('average_q', 0.14706979844385035), ('average_loss', 0), ('n_updates', 0)]
```
