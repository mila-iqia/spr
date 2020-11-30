#!/usr/bin/env bash

python scripts/train_all.py --integrated-model --global-loss --noncontrastive-global-loss --framestack-model --reward-loss-weight 10. --initial_exp_steps 50000

python scripts/train_all.py --integrated-model --global-loss --noncontrastive-global-loss --reward-loss-weight 10. --initial_exp_steps 50000
