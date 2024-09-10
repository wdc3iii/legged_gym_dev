clear; clc;
% Hopper Rec
nms = {
    'eval_hopper_single_int_xkty94y9_1', ...
    'eval_hopper_single_int_eyvpoorf_1', ...
    'eval_hopper_single_int_3j3u2wam_1', ...
    'eval_hopper_single_int_yyu12onw_1', ...
    'eval_hopper_single_int_4oeqlcsm_1', ...
    'eval_hopper_single_int_vscndwrs_1', ...
    'eval_hopper_single_int_3nbdrfdp_1', ...
    'eval_hopper_single_int_c9cbxovu_1', ...
    'eval_hopper_single_int_ks9d6vtv_1', ...
    'eval_hopper_single_int_kyxjirnp_1', ...
    'eval_hopper_single_int_31z4kksi_1' ...
    };

set(groot, 'DefaultAxesFontSize', 17);  % Set default font size for axes labels and ticks
set(groot, 'DefaultTextFontSize', 17);  % Set default font size for text objects
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');  % Set interpreter for axis tick labels
set(groot, 'DefaultTextInterpreter', 'latex');  % Set interpreter for text objects (e.g., titles, labels)
set(groot, 'DefaultLegendInterpreter', 'latex')
set(groot, 'DefaultFigureRenderer', 'painters');
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'DefaultLineMarkerSize', 15)

write_video = false;
nm = nms{1};
load(['data/' nm '.mat']);

%% Visualize
fh = figure(1);
clf;
fh.Position = [1052 697 1750 1113];
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(1);
% subplot(2,2,1)
hold on;
for k = 1:size(z, 1)
    r = norm(z(k, :) - pz_x(k, :));
    tx = z(k, 1); if isnan(tx) tx = 0; end
    ty = z(k, 2); if isnan(ty) ty = 0; end
    rectangle('Position', [tx-r, ty-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', 'k', ...
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

plot(e, 'k.-', DisplayName='$e$');
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

%% Animate
nexttile(2)
up = [0.4940 0.1840 0.5560];
low = [0.9290 0.6940 0.1250];
M = size(nms, 2);
for k = [25, 90, 145]
    for nm_ind = 1:M
        nm = nms{nm_ind};
        c = low * (M - nm_ind) / (M - 1) + up * (nm_ind-1) / (M - 1);
        load(['data/' nm '.mat']);
        
        plot(k+1:k+size(w, 2), w(k, :), Color=c);
    end
end