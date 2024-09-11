function [tls] = plot_cl_trial(nm, fh_num, k)
set(groot, 'DefaultAxesFontSize', 17);  % Set default font size for axes labels and ticks
set(groot, 'DefaultTextFontSize', 17);  % Set default font size for text objects
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');  % Set interpreter for axis tick labels
set(groot, 'DefaultTextInterpreter', 'latex');  % Set interpreter for text objects (e.g., titles, labels)
set(groot, 'DefaultLegendInterpreter', 'latex')
set(groot, 'DefaultFigureRenderer', 'painters');
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'DefaultLineMarkerSize', 15)

load(['data/' nm '.mat']);
disp(['data/' nm '.mat'])

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

if strcmp(k, 'end')
    k = size(z, 1);
end
%% Visualize
fh = figure(fh_num);
clf;
tls = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
subplot(2,1,1)
hold on;
% plot(z_sim(:, 1), z_sim(:, 2), LineWidth=3)

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
    r = max(w(k, j), 0);
    tx = z(k, j, 1); if isnan(tx) tx = 0; end
    ty = z(k, j, 2); if isnan(ty) ty = 0; end
    tube{j} = rectangle('Position', [tx-r, ty-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', 'k', ...
        'LineWidth', 1);
end
% Plot traj
z_line = plot(z(k, :, 1), z(k, :, 2), '.-k', LineWidth=2, Markersize=15);
pz_x_line = plot(pz_x(k, :, 1), pz_x(k, :, 2), '.-', LineWidth=2, Markersize=15);
pz_x_line.Color = "#A2142F";
xlabel('x')
ylabel('y')
xlim(x_lim);
ylim(y_lim);
axis equal

% % Plot error
% subplot(2,2,2)
% hold on
% ind = sum(~isnan(w(k, :)));
% % pred_tube_line = plot([ind:ind + size(w_sol, 2)-2], w_sol(k, 2:end), '--');
% set(gca,'ColorOrderIndex',1)
% tube_line = plot(w(k, :), '.-');
% ek = vecnorm(squeeze(z(k, :, :) - pz_x(k, :, :)), 2, 2);
% e_line = plot(ek, '.-');
% legend('$$w$$', 'tracking error')
% ylabel("Tube")
% xlabel("Node")
% xlim(node_lim)
% ylim(tube_ylim);

% Plot State Trajectories
subplot(2,1,2)
hold on
pm_x_line = plot(squeeze(z(k, :, 1)), '.-', LineWidth=1.5);
% pm_y_line = plot(squeeze(z(k, :, 2)), '.-', LineWidth=1.5);
tr_x_line = plot(squeeze(pz_x(k, :, 1)), '.-', LineWidth=1.5);
% tr_y_line = plot(squeeze(pz_x(k, :, 2)), '.-', LineWidth=1.5);
% legend('planned $x$', 'planned $y$', 'tracking $x$', 'tracking $y$', AutoUpdate='off')
legend('planned $x$', 'tracking $x$', AutoUpdate='off')
ylabel("State")
xlabel("Node")
xlim(node_lim)
ylim(state_y_lim)
yline(0, 'k', LineWidth=0.5)

% subplot(2,2,4)
% hold on;
% input_x_line = plot(squeeze(v(k, :, 1)), '.-');
% input_y_line = plot(squeeze(v(k, :, 2)), '.-');
% legend('$v_x$', '$v_y$', AutoUpdate='off')
% ylabel("Input")
% xlabel("Node")
% xlim(node_lim)
% yline(0, 'k', LineWidth=0.5)
end

