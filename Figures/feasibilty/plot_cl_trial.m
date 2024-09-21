function plot_cl_trial(nm, fh1, fh2, k)
set(groot, 'DefaultAxesFontSize', 17);  % Set default font size for axes labels and ticks
set(groot, 'DefaultTextFontSize', 17);  % Set default font size for text objects
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');  % Set interpreter for axis tick labels
set(groot, 'DefaultTextInterpreter', 'latex');  % Set interpreter for text objects (e.g., titles, labels)
set(groot, 'DefaultLegendInterpreter', 'latex')
set(groot, 'DefaultFigureRenderer', 'painters');
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'DefaultLineMarkerSize', 10)

load(['data/' nm '.mat']);
disp(['data/' nm '.mat'])

x_lim = [-0.1, 1.45];
y_lim = [-0.4, 0.4];
node_lim = [0, size(z, 1) + size(z_sol, 2)];
state_y_lim = [0, 1.5];

if strcmp(k, 'end')
    k = size(z, 1);
end
%% Visualize
figure(fh1);
clf;
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
plot(z(k, :, 1), z(k, :, 2), '.-k', LineWidth=2, Markersize=15);
pz_x_line = plot(pz_x(k, :, 1), pz_x(k, :, 2), '.-', LineWidth=2, Markersize=15);
pz_x_line.Color = "#A2142F";
xlabel('x')
ylabel('y')
axis equal
xlim(x_lim);
ylim(y_lim);

% Plot State Trajectories
figure(fh2)
hold on
plot(squeeze(z(k, :, 1)));
% pm_y_line = plot(squeeze(z(k, :, 2)), '.-', LineWidth=1.5);
plot(squeeze(pz_x(k, :, 1)));
% tr_y_line = plot(squeeze(pz_x(k, :, 2)), '.-', LineWidth=1.5);
% legend('planned $x$', 'planned $y$', 'tracking $x$', 'tracking $y$', AutoUpdate='off')
legend('planned $x$', 'tracking $x$', AutoUpdate='off')
ylabel("State")
xlabel("Node")
xlim(node_lim)
ylim(state_y_lim)
yline(0, 'k', LineWidth=0.5)
end

