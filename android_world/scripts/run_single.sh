MODEL_NAME="gui_owl"
MODEL="GUI-Owl-7B"

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
TRAJ_OUTPUT_PATH="results/"$MODEL"/traj_"$current_time
LOG=$TRAJ_OUTPUT_PATH"/log_"$current_time".log"
#TRAJ_OUTPUT_PATH="./results/GUI-Owl-7B/traj_2025-11-25_18-00-58"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$TRAJ_OUTPUT_PATH"

python minimal_task_runner.py \
  --task=ExpenseDeleteDuplicates \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key="token-abc123" \
  --base_url="http://127.0.0.1:8000/v1" \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --console_port=5554 2>&1 | tee "$LOG" \
  #--perform_emulator_setup | tee "$LOG"