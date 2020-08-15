终于折腾好了在服务器上跑起来baseline.
完整命令:
nvidia-docker run --net=host --volume="$HOME/workspace:/tf" --user 0:0 -it chenqibin422/minerl bash

但是完整跑完时间较久，所以只跑了几个episode.

baseline1: ppo

```
INFO     - 2020-08-15 06:01:56,887 - [chainerrl.experiments.train_agent train_agent 59] outdir:results/MineRLTreechop-v0/ppo/20200815T054120.495466 step:4000 episode:1 R:8.0
INFO     - 2020-08-15 06:01:56,888 - [chainerrl.experiments.train_agent train_agent 60] statistics:[('average_value', 0.025590062258881517), ('average_entropy', 1.538290573477745), ('average_value_loss', 0.05641872310661711), ('average_policy_loss', -0.07323072407598374), ('n_updates', 96), ('explained_variance', 0.13813830775879976)]
```
