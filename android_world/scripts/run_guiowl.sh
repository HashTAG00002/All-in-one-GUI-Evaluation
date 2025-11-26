export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=none
export GLOG_minloglevel=1  # 1=WARNING, 2=ERROR, 3=FATAL
export GLOG_v=0
export GLOG_logtostderr=1

MODEL_NAME="gui_owl"
MODEL="GUI-Owl-7B"

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
TRAJ_OUTPUT_PATH="results/"$MODEL"/traj_"$current_time
LOG=$TRAJ_OUTPUT_PATH"/log_"$current_time".log"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$TRAJ_OUTPUT_PATH"

python run_ma3.py \
  --suite_family=android_world \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key="token-abc123" \
  --base_url="http://127.0.0.1:8000/v1" \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --grpc_port=8554 \
  --console_port=5554 2>&1 | tee "$LOG" \
  #--perform_emulator_setup | tee "$LOG"
  