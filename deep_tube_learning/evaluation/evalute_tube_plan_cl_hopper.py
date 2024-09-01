import subprocess

__conda_setup="$('/home/wcompton/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# Start server process
server_process = subprocess.Popen([
    'bash', '-c',
    '__conda_setup=\"$(\'/home/wcompton/miniforge3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)\" && eval \"$__conda_setup\" && source /home/wcompton/miniforge3/etc/profile.d/mamba.sh && mamba activate deep_tubes_dev && python evaluate_tube_plan_cl_server.py'
])

# Start client process
client_process = subprocess.Popen([
    'bash', '-c',
    '__conda_setup=\"$(\'/home/wcompton/miniforge3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)\" && eval \"$__conda_setup\" && source /home/wcompton/miniforge3/etc/profile.d/mamba.sh && mamba activate legged_gym_dev && python evaluate_tube_plan_cl_client.py'
])

# Optionally wait for processes to finish
server_process.wait()
client_process.wait()