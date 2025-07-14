#!/usr/bin/env python3
"""
检查checkpoint文件路径的脚本
"""

import os
from legged_gym.utils import get_args, task_registry, get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR

def check_checkpoint():
    args = get_args()
    
    # 获取配置
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 设置参数
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    
    # 构建路径
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    
    print(f"Task: {args.task}")
    print(f"Experiment name: {train_cfg.runner.experiment_name}")
    print(f"Log root: {log_root}")
    print(f"Load run: {train_cfg.runner.load_run}")
    print(f"Checkpoint: {train_cfg.runner.checkpoint}")
    
    try:
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        print(f"Resume path: {resume_path}")
        
        if os.path.exists(resume_path):
            print("✓ Checkpoint file exists!")
            # 尝试加载checkpoint以验证内容
            import torch
            loaded_dict = torch.load(resume_path, map_location='cpu')
            print(f"✓ Checkpoint loaded successfully!")
            print(f"✓ Saved iteration: {loaded_dict.get('iter', 'Not found')}")
        else:
            print("✗ Checkpoint file does not exist!")
            
            # 列出可用的runs
            print("\nAvailable runs:")
            if os.path.exists(log_root):
                runs = os.listdir(log_root)
                runs.sort()
                for run in runs:
                    if run != 'exported':
                        print(f"  - {run}")
            else:
                print(f"  Log root directory doesn't exist: {log_root}")
                
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == '__main__':
    check_checkpoint() 