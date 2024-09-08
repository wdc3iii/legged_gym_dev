clear; clc;
% Hopper Rec
% nm = 'eval_hopper_single_int_sjiqi49f_1';
% nm = 'eval_hopper_single_int_7p1zump7_1';
% nm = 'eval_hopper_single_int_msj97p19_1';
% nm = 'eval_hopper_single_int_ks1eg0xw_1';
% nm = 'eval_hopper_single_int_rg5itafm_1';
% nm = 'eval_hopper_single_int_j88i9kim_1';
% nm = 'eval_hopper_single_int_mnfs3r5v_1';
% nm = 'eval_hopper_single_int_r5xu847t_1';
% nm = 'eval_hopper_single_int_l4wnnx72_1';
% nm = 'eval_hopper_single_int_w8flp57h_1';
% nm = 'eval_hopper_single_int_f9zr70ds_1';

% Hopper OS
% nm = 'eval_hopper_single_int_dy9ivoc2_1';
% nm = 'eval_hopper_single_int_kma11ykr_1';
% nm = 'eval_hopper_single_int_i8bev8qs_1';
% nm = 'eval_hopper_single_int_743mwobx_1';
% nm = 'eval_hopper_single_int_q9r1jp40_1';
% nm = 'eval_hopper_single_int_55xbfsa8_1';
% nm = 'eval_hopper_single_int_xstep89m_1';
% nm = 'eval_hopper_single_int_r0hxk04w_1';
% nm = 'eval_hopper_single_int_ho9ss350_1';
% nm = 'eval_hopper_single_int_y7udluc0_1';
% nm = 'eval_hopper_single_int_1rf51hpn_1';

% Double Rec
% nm = 'eval_double_single_int_mrw2ieou_1';
% nm = 'eval_double_single_int_vugufzon_1';
% nm = 'eval_double_single_int_28fyddeg_1';
% nm = 'eval_double_single_int_bzcnvzc8_1';
% nm = 'eval_double_single_int_nxwo1usp_1';
% nm = 'eval_double_single_int_2p3relpc_1';
% nm = 'eval_double_single_int_umfjf3uu_1';
% nm = 'eval_double_single_int_thsm2yqh_1';
% nm = 'eval_double_single_int_g7yvoy73_1';
% nm = 'eval_double_single_int_wn7ag3y4_1';
nm = 'eval_double_single_int_42cp4zi6_1';

% Double OS
% nm = 'eval_double_single_int_d2rcffmp_1';
% nm = 'eval_double_single_int_zsjzccbb_1';
% nm = 'eval_double_single_int_4tcr7hfa_1';
% nm = 'eval_double_single_int_6w6s1don_1';
% nm = 'eval_double_single_int_bte0x966_1';
% nm = 'eval_double_single_int_4zawox51_1';
% nm = 'eval_double_single_int_syqvx07n_1';
% nm = 'eval_double_single_int_6gn5gan2_1';
% nm = 'eval_double_single_int_a65vb9rm_1';
% nm = 'eval_double_single_int_adeuul8f_1';
% nm = 'eval_double_single_int_3iy8gtai_1';

set(groot, 'DefaultAxesFontSize', 17);  % Set default font size for axes labels and ticks
set(groot, 'DefaultTextFontSize', 17);  % Set default font size for text objects
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');  % Set interpreter for axis tick labels
set(groot, 'DefaultTextInterpreter', 'latex');  % Set interpreter for text objects (e.g., titles, labels)
set(groot, 'DefaultLegendInterpreter', 'latex')
set(groot, 'DefaultFigureRenderer', 'painters');
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'DefaultLineMarkerSize', 15)

write_video = false;
load(['data/' nm '.mat']);

%% Visualize
fh = figure(1);
clf;
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(1);
% subplot(2,2,1)
hold on;

% Plot traj
tube = zeros(size(w, 1));
tube(1) = e(1);
tube_plt = cell(size(w, 1), 1);
for j = 1:size(w, 1)
    r = tube(j);
    tx = z(j, 1); if isnan(tx) tx = 0; end
    ty = z(j, 2); if isnan(ty) ty = 0; end
    tube_plt{j} = rectangle('Position', [tx-r, ty-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', 'k', ...
        'LineWidth', 1);
end
% Plot prediction
sol_tube = cell(size(w, 2), 1);
for j = 1:size(w, 2)
    r = max(w(1, j), 0);
    sol_tube{j} = rectangle('Position', [z(j + 1, 1)-r, z(j + 1, 2)-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', "#77AC30", ...
        'LineWidth', 1);
end
plot(z(:, 1), z(:, 2), '.-k', LineWidth=2, Markersize=15);
pz_x_line = plot(pz_x(:, 1), pz_x(:, 2), '.-', LineWidth=2, Markersize=15);
pz_x_line.Color = "#A2142F";
plot(z(1, 1), z(1, 2), 'go')
plot(pz_x(1, 1), pz_x(1, 2), 'go')
xlabel('x')
ylabel('y')
axis equal

% Plot error
% subplot(2,2,2)
nexttile;
hold on
tube_line = plot(1, tube(1));
set(gca,'ColorOrderIndex',1)
pred_tube_line = plot(2:size(w, 2) + 1, w(1, :), '--');

plot(e, '.-');
legend('$w$', '$w$ predicted', 'tracking error')
ylabel("Tube")
xlabel("Node")
node_lim = [0 size(z, 1)];
xlim(node_lim)

% Plot State Trajectories
% subplot(2,2,3)
nexttile;
hold on
plot(z(:, 1), '.-', LineWidth=1.5);
plot(z(:, 2), '.-', LineWidth=1.5);
plot(pz_x(:, 1), '.-', LineWidth=1.5);
plot(pz_x(:, 2), '.-', LineWidth=1.5);
legend('$x$', '$y$', 'tracking $x$', 'tracking $y$', AutoUpdate='off')
yline(0, 'k', LineWidth=0.5)
ylabel("State")
xlabel("Node")
xlim(node_lim)
% subplot(2,2,4)
nexttile;
hold on;
plot(v(:, 1), '.-', LineWidth=1.5);
plot(v(:, 2), '.-', LineWidth=1.5);
legend('$v_x$', '$v_y$', AutoUpdate='off')
yline(0, 'k', LineWidth=0.5)
ylabel("Input")
xlabel("Node")
xlim(node_lim)
% subplot(2, 2, 1)
nexttile(1);

%% Animate
if write_video
    video = VideoWriter(nm); % Create the VideoWriter object for MP4 format
    video.FrameRate = 60; % Set the frame rate (optional)
    video_dt = 1/video.FrameRate;
    open(video);
end

for k = 1:size(w, 1)
    tic
    for j = 1:size(w, 1)
        r = max(tube(j), 0);
        tx = z(j, 1); if isnan(tx) tx = 0; end
        ty = z(j, 2); if isnan(ty) ty = 0; end
        tube_plt{j}.Position = [tx-r, ty-r, 2*r, 2*r];
    end
    tube(k + 1) = w(k, 1);
    tube_line.XData = [1:k];
    tube_line.YData = tube(1:k);
    pred_tube_line.XData = [k+1:k+size(w, 2)];
    pred_tube_line.YData = w(k, :);
    % Plot prediction
    for j = 1:size(w, 2)
        r = max(w(k, j), 0);
        sol_tube{j}.Position = [z(k + j, 1)-r, z(k + j, 2)-r, 2*r, 2*r];
    end
    nexttile(1);
    title_str = sprintf('Time Step:  %d', k);
    title(title_str, 'Interpreter', 'none');
    drawnow

    % Write the frame to the video
    if write_video
        frame = getframe(gcf);
        for ii = 1:round(0.1 / video_dt)
            writeVideo(video, frame);
        end
    end
    if toc < 0.1
        pause(0.1 - toc);
    end
    % pause
end

if write_video
    close(video);
    disp(['Video saved as ', videoFileName]);
end