# SOPD

SOPD means Singleton OPD. It reuses the COPD OPD guidance path, but applies the
teacher signal only to steps whose GiGPO state group has size 1.

The goal is to add fine-grained on-policy distillation signal where state-group
relative advantage has no comparison partner, while leaving multi-sample
state groups to the usual GiGPO/COPD relative signal. The WebShop script keeps
GiGPO-style step advantage on by setting `algorithm.copd.step_advantage_w=1.0`.

SOPD treats all singleton candidate steps as key training steps. In
`algorithm.copd.singleton_only=True` mode, failed-only filtering is ignored so
every singleton step can receive OPD guidance after successful analysis.
SOPD uses step hints only; episode-hint teacher weight is fixed to 0.0 in the
WebShop launch script.
