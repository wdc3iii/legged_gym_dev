from trajopt.rom_dynamics import SingleInt2D
import torch
from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *
from trajopt.rom_dynamics import DoubleInt2D


# prob_str = 'right'
# prob_str = 'right_wide'
prob_str = 'gap'

track_warm = True

# warm_start = 'start'
# warm_start = 'goal'
# warm_start = 'interpolate'
warm_start = 'nominal'

# tube_ws = 0
tube_ws = 0.5

# tube_dyn = 'l1'
# tube_dyn = "l2"
# tube_dyn = "l1_rolling"
# tube_dyn = "l2_rolling"
tube_dyn = "NN_oneshot"
nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/0ks4vit2"  # 128x128 softplus b=5

Kp = 10
Kd = 10

def eval_model():
    # Experiment whose model to evaluate
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/l0oh8o1b"  # 32x32
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/k1kfktrl"   # 128x128
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/3vdx800j"   # 256x256
    exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/0ks4vit2"    # 128x128 Softplus b=5
    model_name = f'{exp_name}_model:best'



    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)

    start, goal, obs, vel_max, pos_max, dt = problem_dict[prob_str]["start"], problem_dict[prob_str]["goal"], problem_dict[prob_str]["obs"], \
                                             problem_dict[prob_str]["vel_max"], problem_dict[prob_str]["pos_max"], problem_dict[prob_str]["dt"]

    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    planning_model = CasadiSingleInt2D(dt, -z_max, z_max, -v_max, v_max)

    Q = 10 * np.eye(2)
    Qw = 0
    R = 10 * np.eye(2)
    N = 50
    w_max = 1

    double_z_max = torch.tensor([np.inf, np.inf, 2., 2.])
    double_v_max = torch.tensor([2., 2.])
    double_int = DoubleInt2D(dt, -double_z_max, double_z_max, -double_v_max, double_v_max, n_robots=1, backend='numpy')

    tube_dynamics = get_tube_dynamics(tube_dyn, nn_path=nn_path)

    tube_ws_str = str(tube_ws).replace('.', '_')
    fn = f"data/tube_{prob_str}_{warm_start}_{tube_dyn}_{tube_ws_str}.csv"
    sol, solver = solve_tube(start, goal, obs, planning_model, tube_dynamics, N, Q, Qw, R, w_max, warm_start=warm_start,
                             tube_ws=tube_ws, debug_filename=fn, max_iter=1000, track_warm=track_warm)

    z_sol, v_sol, w_sol = extract_solution(sol, N, planning_model.n, planning_model.m)

    x = np.zeros((1, z_sol.shape[0], double_int.n))
    u = np.zeros((1, v_sol.shape[0], double_int.m))
    pz_x = np.zeros_like(z_sol)

    for t in range(v_sol.shape[0]):
        # Decide fom action
        xt = x[:, t, :]
        zt = z_sol[t, :]
        # zt_p = traj_gen.trajectory[:, 1, :]
        vt_p = v_sol[min(t + 1, v_sol.shape[0] - 1), :]
        ut = double_int.clip_v_z(xt, Kp * (zt - xt[:, :2]) + Kd * (vt_p - xt[:, 2:]))

        xt_p1 = double_int.f(xt, ut.numpy())

        # Store
        x[:, t + 1, :] = xt_p1
        u[:, t, :] = ut
        pz_x[t + 1, :] = double_int.proj_z(xt_p1)

    w = np.linalg.norm(z_sol - pz_x, axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.concatenate((w[0, None], v_sol.reshape((-1,))))[None, :]
    model = instantiate(model_cfg.model)(data.shape[-1], v_sol.shape[0])
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"alpha: {model_cfg.loss.alpha}")

    with torch.no_grad():
        fw = model(torch.from_numpy(data).float().to(device)).cpu().detach().numpy()
        fw = np.concatenate([w[0, None, None], fw], axis=-1).squeeze()

        plt.figure()
        err = fw - w
        succ_rate = np.mean(err >= 0)
        print(succ_rate)
        plt.plot(err)
        plt.axhline(0, c='k')
        plt.xlabel('Time')
        plt.ylabel('Tube Error')
        plt.show()

        plt.figure()
        plt.plot(fw, 'k', label='Tube Error')
        plt.plot(w, 'b', label='Normed Error')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title("OneShot Tube Bounds")
        plt.legend()
        plt.savefig('data/OneShotTubeBound.png')
        plt.show()

        fig, ax = plt.subplots()
        plot_problem(ax, obs, start, goal)
        SingleInt2D.plot_tube(ax, z_sol, fw[:, None])
        SingleInt2D.plot_spacial(ax, z_sol, 'k.-')
        SingleInt2D.plot_spacial(ax, pz_x, 'b.-')
        plt.axis("square")
        plt.title("OneShot Tube")
        plt.show()
    print(f"Total Success Rate: {succ_rate}")


if __name__ == "__main__":
    eval_model()
