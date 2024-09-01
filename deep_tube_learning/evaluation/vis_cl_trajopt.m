clear; clc;
% Double Int N = 50 (model training horizon)
% nm = 'cl_tube_complex_07uwnu78_nominal_Rv_10_10_NN_recursive_evaluate_True'; % N = 50 MPC horizon 20x realtime
% nm = 'cl_tube_complex_07uwnu78_nominal_Rv_10_10_N_25_NN_recursive_evaluate_True'; % N = 25 MPC horizon 10x realtime
% nm = 'cl_tube_complex_07uwnu78_nominal_Rv_10_10_N_25_dk_2_NN_recursive_evaluate_True'; % N = 25, dk = 2
% nm = 'cl_tube_complex_07uwnu78_nominal_Rv_10_10_N_10_NN_recursive_evaluate_True'; % N = 10 MPC horizon FAILS

% Hopper N = 50 (model training horizon)
% nm = 'cl_tube_complex_rkm53z6t_nominal_Rv_10_10_NN_recursive_evaluate_True';
% nm = 'cl_tube_gap_rkm53z6t_nominal_Rv_10_10_NN_recursive_evaluate_True';
nm = 'cl_tube_gap_rkm53z6t_nominal_Rv_10_10_N_25_dk_2_NN_recursive_evaluate_True';

% Hopper N = 10 (model training horizon)
% nm = 'cl_tube_complex_nqkkk3af_nominal_Rv_10_10_NN_recursive_evaluate_True';
% nm = 'cl_tube_gap_nqkkk3af_nominal_Rv_10_10_NN_recursive_evaluate_True';

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

max_tube = max(w, [], 'all');
x_lim = [
    min([min(z(:, :, 1), [], 'all'), min(pz_x(:, :, 1), [], 'all'), min(obs_x)]) - max_tube - 0.1, ...
    max([max(z(:, :, 1), [], 'all'), min(pz_x(:, :, 1), [], 'all'), max(obs_x)]) + max_tube + 0.1
];
y_lim = [
    min([min(z(:, :, 2), [], 'all'), min(pz_x(:, :, 2), [], 'all'), min(obs_y)]) - max_tube - 0.1, ...
    max([max(z(:, :, 2), [], 'all'), min(pz_x(:, :, 2), [], 'all'), max(obs_y)]) + max_tube + 0.1
];
node_lim = [0, size(z, 1) + size(z_sol, 2)];
tube_ylim = [0, max([max(w, [], 'all'), max(w_sol, [], 'all'), max(vecnorm(z - pz_x, 2, 3), [], 'all')]) * 1.05];
state_y_lim = [
    min([min(z, [], 'all'), min(z_sol, [], 'all'), min(pz_x, [], 'all')]), ...
    max([max(z, [], 'all'), max(z_sol, [], 'all'), max(pz_x, [], 'all')])
];
input_y_lim = [
    min(min(v, [], 'all'), min(v_sol, [], 'all')), ...
    max(max(v, [], 'all'), max(v_sol, [], 'all'))
];
state_y_lim = [state_y_lim(1) - diff(state_y_lim) * 0.05, state_y_lim(2) + diff(state_y_lim) * 0.05];
input_y_lim = [input_y_lim(1) - diff(input_y_lim) * 0.05, input_y_lim(2) + diff(input_y_lim) * 0.05];
%% Visualize
fh = figure(1);
clf;
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(1);
% subplot(2,2,1)
hold on;

% Plot problem
plot(z0(1), z0(2), 'go')
plot(zf(1), zf(2), 'rx')
for ii = 1:size(obs_r, 2)
    r = obs_r(ii); x = obs_x(ii); y = obs_y(ii) ;
    rectangle('Position', [x-r, y-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', 'r', ...
        'FaceColor', 'r', ...
        'LineWidth', 1);
end

tube = cell(size(w, 2), 1);
for j = 1:size(w, 2)
    r = max(w(1, j), 0);
    tx = z(1, j, 1); if isnan(tx) tx = 0; end
    ty = z(1, j, 2); if isnan(ty) ty = 0; end
    tube{j} = rectangle('Position', [tx-r, ty-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', 'k', ...
        'LineWidth', 1);
end
% Plot prediction
z_sol_line = plot(z_sol(1, :, 1), z_sol(1, :, 2), '.-k', LineWidth=2, Markersize=15);
z_sol_line.Color = "#77AC30";
sol_tube = cell(size(w, 2) - 1, 1);
for j = 1:size(w_sol, 2)
    r = max(w_sol(1, j), 0);
    sol_tube{j} = rectangle('Position', [z_sol(1, j+1, 1)-r, z_sol(1, j+1, 2)-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', "#77AC30", ...
        'LineWidth', 1);
end
% Plot traj
z_line = plot(z(1, :, 1), z(1, :, 2), '.-k', LineWidth=2, Markersize=15);
pz_x_line = plot(pz_x(1, :, 1), pz_x(1, :, 2), '.-', LineWidth=2, Markersize=15);
pz_x_line.Color = "#A2142F";
xlabel('x')
ylabel('y')
xlim(x_lim);
ylim(y_lim);
axis equal

% Plot error
% subplot(2,2,2)
nexttile;
hold on
ind = sum(~isnan(w(1, :)));
pred_tube_line = plot([ind:ind + size(w_sol, 2)-2], w_sol(1, 2:end), '--');
set(gca,'ColorOrderIndex',1)
tube_line = plot(w(1, :), '.-');
ek = vecnorm(squeeze(z(1, :, :) - pz_x(1, :, :)), 2, 2);
e_line = plot(ek, '.-');
legend('$w$ predicted', '$$w$$', 'tracking error')
ylabel("Tube")
xlabel("Node")
xlim(node_lim)
ylim(tube_ylim);
% gv_dyn_line = plot(cv0.Dynamics, '.-', DisplayName="Dynamics");
% gv_obs_lines = cell(numel(obs_r), 1);
% for ii = 1:numel(obs_r)
%     gv_obs_line{ii} = plot(cv0.(['Obstacle ' num2str(ii - 1)]), '.-', DisplayName=['Obstacle ' num2str(ii)]);
% end
% gv_tub_line = plot(cv0.('Tube Dynamics'), '.-', DisplayName="Tube");
% gv_ic_line = plot(cv0.('Initial Condition'), '.-', DisplayName="IC");
% legend()
% title("Constraint Violation")

% Plot State Trajectories
% subplot(2,2,3)
nexttile;
hold on
pred_x_line = plot([ind-1:ind + size(z_sol, 2)-2], squeeze(z_sol(1, :, 1)), '--');
pred_y_line = plot([ind-1:ind + size(z_sol, 2)-2], squeeze(z_sol(1, :, 2)), '--');
set(gca,'ColorOrderIndex',1)
pm_x_line = plot(squeeze(z(1, :, 1)), '.-', LineWidth=1.5);
pm_y_line = plot(squeeze(z(1, :, 2)), '.-', LineWidth=1.5);
tr_x_line = plot(squeeze(pz_x(1, :, 1)), '.-', LineWidth=1.5);
tr_y_line = plot(squeeze(pz_x(1, :, 2)), '.-', LineWidth=1.5);
legend('$x$ pred', '$y$ pred', 'planned $x$', 'planned $y$', 'tracking $x$', 'tracking $y$', AutoUpdate='off')
ylabel("State")
xlabel("Node")
xlim(node_lim)
ylim(state_y_lim)
yline(0, 'k', LineWidth=0.5)

% subplot(2,2,4)
nexttile;
hold on;
pred_input_x_line = plot([ind-1:ind + size(v_sol, 2)-2], squeeze(v_sol(1, :, 1)), '--');
pred_input_y_line = plot([ind-1:ind + size(v_sol, 2)-2], squeeze(v_sol(1, :, 2)), '--');
set(gca,'ColorOrderIndex',1)
input_x_line = plot(squeeze(v(1, :, 1)), '.-');
input_y_line = plot(squeeze(v(1, :, 2)), '.-');
legend('pred $v_x$', 'pred $v_y$', '$v_x$', '$v_y$', AutoUpdate='off')
ylabel("Input")
xlabel("Node")
xlim(node_lim)
yline(0, 'k', LineWidth=0.5)

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
    z_line.XData = z(k, :, 1);
    z_line.YData = z(k, :, 2);
    pz_x_line.XData = pz_x(k, :, 1);
    pz_x_line.YData = pz_x(k, :, 2);
    for j = 1:size(w, 2)
        r = max(w(k, j), 0);
        tx = z(k, j, 1); if isnan(tx) tx = 0; r = 0;end
        ty = z(k, j, 2); if isnan(ty) ty = 0; end
        tube{j}.Position = [tx-r, ty-r, 2*r, 2*r];
    end
    % Plot prediction
    z_sol_line.XData = z_sol(k, :, 1);
    z_sol_line.YData = z_sol(k, :, 2);
    for j = 1:size(w_sol, 2)
        r = max(w_sol(k, j), 0);
        sol_tube{j}.Position = [z_sol(k, j, 1)-r, z_sol(k, j, 2)-r, 2*r, 2*r];
    end
    % cv = eval(['cv' num2str(k - 1)]);
    % gv_dyn_line.YData = cv.Dynamics;
    % for ii = 1:numel(obs_r)
    %     gv_obs_line{ii}.YData = cv.(['Obstacle ' num2str(ii - 1)]);
    % end
    % gv_tub_line.YData = cv.('Tube Dynamics');
    % gv_ic_line.YData = cv.('Initial Condition');
    e_line.YData = vecnorm(squeeze(z(k, :, :) - pz_x(k, :, :)), 2, 2);
    ind = sum(~isnan(w(k, :)));
    inds = [ind-1:ind + size(z_sol, 2)-2];

    pred_tube_line.XData = inds(2:end);
    pred_tube_line.YData = squeeze(w_sol(k, 1:end));

    pred_x_line.XData = inds;
    pred_x_line.YData = squeeze(z_sol(k, :, 1));
    pred_y_line.XData = inds;
    pred_y_line.YData = squeeze(z_sol(k, :, 2));
    pm_x_line.YData = squeeze(z(k, :, 1));
    pm_y_line.YData = squeeze(z(k, :, 2));
    tr_x_line.YData = squeeze(pz_x(k, :, 1));
    tr_y_line.YData = squeeze(pz_x(k, :, 2));

    pred_input_x_line.XData = inds(1:end-1);
    pred_input_x_line.YData = squeeze(v_sol(k, :, 1));
    pred_input_y_line.XData = inds(1:end-1);
    pred_input_y_line.YData = squeeze(v_sol(k, :, 2));

    input_x_line.YData = squeeze(v(k, :, 1));
    input_y_line.YData = squeeze(v(k, :, 2));
    tube_line.YData = w(k, :);
    if k > 1
        dt = (timing(k) - timing(k - 1)) * 1e-9;
    else
        dt = timing(k) * 1e-9;
    end
    if dt <= 0.1
        bs = 'True';
    else
        bs = 'False';
    end
    nexttile(1);
    title_str = sprintf('Time Step:  %d      Real Time:  %.1fs     Compute Time:  %.2fs      Iterations Realtime:  %s', ...
                   k, 0.1 * k, timing(k) * 1e-9, bs);
    title(title_str, 'Interpreter', 'none');
    xlim(x_lim);
    ylim(y_lim);
    nexttile(2);
    xlim(node_lim);
    ylim(tube_ylim);
    nexttile(3);
    xlim(node_lim);
    ylim(state_y_lim);
    nexttile(4);
    xlim(node_lim);
    ylim(input_y_lim);
    drawnow

    % Write the frame to the video
    if write_video
        frame = getframe(gcf);
        for ii = 1:round(0.1 / video_dt)
            writeVideo(video, frame);
        end
    end
    if toc - tic > 0
        pause(toc - tic)
    end
end

if write_video
    close(video);
    disp(['Video saved as ', videoFileName]);
end    