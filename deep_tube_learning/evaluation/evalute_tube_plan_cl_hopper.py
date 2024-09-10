import subprocess

__conda_setup="$('/home/wcompton/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

exp_names = [  # Hopper Recursive
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/sjiqi49f",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/7p1zump7",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/msj97p19",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/ks1eg0xw",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/rg5itafm",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/j88i9kim",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/mnfs3r5v",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/r5xu847t",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/l4wnnx72",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/w8flp57h",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/f9zr70ds"
]

# exp_names = [  # Hopper OS
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/dy9ivoc2",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/kma11ykr",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/i8bev8qs",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/743mwobx",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/q9r1jp40",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/55xbfsa8",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/xstep89m",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/r0hxk04w",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/ho9ss350",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/y7udluc0",
#     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/1rf51hpn"
# ]

for exp_name in exp_names:
    # Start server process
    server_process = subprocess.Popen([
        'bash', '-c',
        f'__conda_setup=\"$(\'/home/wcompton/miniforge3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)\" && eval \"$__conda_setup\" && source /home/wcompton/miniforge3/etc/profile.d/mamba.sh && mamba activate deep_tubes_dev && export SNOPT_LICENSE=/home/wcompton/Repos/snopt/snopt7.lic && python evaluate_tube_plan_cl_server.py \"{exp_name}\"'
    ])

    # Start client process
    client_process = subprocess.Popen([
        'bash', '-c',
        '__conda_setup=\"$(\'/home/wcompton/miniforge3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)\" && eval \"$__conda_setup\" && source /home/wcompton/miniforge3/etc/profile.d/mamba.sh && mamba activate legged_gym_dev && python evaluate_tube_plan_cl_client.py'
    ])

    # Optionally wait for processes to finish
    server_process.wait()
    client_process.wait()