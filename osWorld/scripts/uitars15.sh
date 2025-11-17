python run_multienv_uitars15_v1.py \
    --model UI-TARS-1.5-7B \
    --headless \
    --observation_type screenshot \
    --action_space pyautogui \
    --test_all_meta_path evaluation_examples/test_nogdrive.json \
    --max_steps 50 \
    --num_envs 13 \
    --provider_name aws \
    --region ap-east-1 \
    --client_password osworld-public-evaluation \
    --result_dir ./results_uitars15
    
