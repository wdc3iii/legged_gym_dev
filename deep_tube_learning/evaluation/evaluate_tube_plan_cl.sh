#!/bin/bash

# Initialize mamba for the script shell
source /home/wcompton/miniforge3/etc/profile.d/mamba.sh

# Start the server in a new tmux session with proper mamba initialization
echo "Starting server (Optimization routine)"
tmux new-session -d -s optimization_server "bash 'source /home/wcompton/miniforge3/etc/profile.d/mamba.sh; mamba activate deep_tubes_dev; export PYTHONPATH=~/Repos/legged_gym_dev:\$PYTHONPATH; python ~/Repos/legged_gym_dev/deep_tube_learning/evaluation/evaluate_tube_plan_cl_server.py' > server_log.txt 2>&1"

# Check tmux sessions to confirm the server started
tmux ls

# Wait a bit to ensure the server starts
sleep 2
tmux ls

# Start the client in a new tmux session with proper mamba initialization
echo "Starting Client (simulation)"
tmux new-session -d -s optimization_client "bash 'source /home/wcompton/miniforge3/etc/profile.d/mamba.sh; mamba activate legged_gym_dev; export PYTHONPATH=~/Repos/legged_gym_dev:\$PYTHONPATH; python ~/Repos/legged_gym_dev/deep_tube_learning/evaluation/evaluate_tube_plan_cl_client.py' > client_log.txt 2>&1"

# Check tmux sessions to confirm the client started
tmux ls
sleep 2
tmux ls

# Optional: Attach to tmux session to see outputs
# tmux attach-session -t optimization_server