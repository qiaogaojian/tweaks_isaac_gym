# sudo code . --user-data-dir=/root/.vscode
# python humanoid/scripts/train.py --task=pai_ppo --run_name v1 --num_envs 100
# wandb: 5cd7d9e05d45617d4cebeaca4458b48688e4a5f5
import sys

sys.path.append('/mnt/workspace/git/livelybot_rl_control/venv/lib/python3.8/site-packages')

num = 123
print(f"Hello {num}")
