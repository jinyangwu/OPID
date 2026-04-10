while true; do
    # 获取 GPU 0 的显存使用量 (如果你是其他卡，修改 --id=0)
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=4)
    # 判断显存是否小于 1000 MB
    if [ "$mem_used" -lt 4000 ]; then
        echo ">>> GPU 当前空闲 (${mem_used}MB)，立即启动脚本！"
        
        # --- 这里替换成你的真实脚本路径 ---
        # bash examples/gtpo_trainer/ys_run_alfworld_branch_cs_bsz16_wo_p.sh
        # bash scripts/cold_start/all_1.5b_4.sh
        # bash examples/gtpo_trainer/ys_run_alfworld_branch_cs_bsz16_wo_p.sh
        # bash scripts/cold_start/all_7b_4.sh
        bash examples/grpo_trainer/run_webshop_copd.sh
        # bash examples/gtpo_trainer/7b_ys_run_alfworld_branch_cs_bsz16_wo_p_L2.sh
        break  # 跑完脚本跳出循环
    else
        echo "GPU 忙碌中 (${mem_used}MB)，等待 60 秒后重试..."
        sleep 120
    fi
done