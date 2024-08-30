% tbl = readtable("tube_right_wide_nominal_l2_0.csv");
% tbl = readtable("tube_gap_nominal_l2_0.csv");
% tbl = readtable("tube_right_nominal_l2_0.csv");
% tbl = readtable("tube_right_wide_nominal_l1_0.csv");
% tbl = readtable("tube_gap_nominal_l1_0.csv");
tbl = readtable("tube_gap_nominal_NN_oneshot_evaluate_False.csv");
% tbl = readtable("tube_gap_nominal_l1_rolling_0.csv");
% tbl = readtable("tube_right_nominal_l1_0.csv");
% tbl = readtable("tube_gap_nominal_NN_oneshot_evaluate.csv");
% tbl = readtable("tube_right_wide_nominal_NN_oneshot_evaluate.csv");


cols = tbl.Properties.VariableNames;

iters = tbl.iter;

z_cols = find(cellfun(@(x) contains(x, 'z_') && ~contains(x, 'lb') && ~contains(x, 'ub') && ~contains(x, 'ic') && ~contains(x, '_g_'), cols));
v_cols = find(cellfun(@(x) contains(x, 'v_') && ~contains(x, 'lb') && ~contains(x, 'ub') && ~contains(x, 'ic')&& ~contains(x, 'prev'), cols));
w_cols = find(cellfun(@(x) contains(x, 'w_') && ~contains(x, 'lb') && ~contains(x, 'ub') && ~contains(x, 'ic'), cols));
z = tbl{:, z_cols};
z = reshape(z, size(z, 1), size(z, 2) / 2, 2);

v = tbl{:, v_cols};
v = reshape(v, size(v, 1), size(v, 2) / 2, 2);

w = tbl{:, w_cols};

dyn_cols = find(cellfun(@(x) contains(x, 'dyn_') && ~contains(x, 'lb_') && ~contains(x, 'ub_'), cols));
g_dyn = tbl{:, dyn_cols};
dyn_cols_lb = find(cellfun(@(x) contains(x, 'lb_dyn_'), cols));
g_dyn_lb = tbl{1, dyn_cols_lb};
dyn_cols_ub = find(cellfun(@(x) contains(x, 'ub_dyn_'), cols));
g_dyn_ub = tbl{1, dyn_cols_ub};

tube_cols = find(cellfun(@(x) contains(x, 'tube_') && ~contains(x, 'lb_') && ~contains(x, 'ub_'), cols));
g_tube = tbl{:, tube_cols};
tube_cols_lb = find(cellfun(@(x) contains(x, 'lb_tube_'), cols));
g_tube_lb = tbl{1, tube_cols_lb};
tube_cols_ub = find(cellfun(@(x) contains(x, 'ub_tube_'), cols));
g_tube_ub = tbl{1, tube_cols_ub};

ic_cols = find(cellfun(@(x) (contains(x, 'ic_x') || contains(x, 'ic_y')) && ~contains(x, 'lb_') && ~contains(x, 'ub_'), cols));
g_ic = tbl{:, ic_cols};
ic_cols_lb = find(cellfun(@(x) contains(x, 'lb_ic_x') || contains(x, 'lb_ic_y'), cols));
g_ic_lb = tbl{1, ic_cols_lb};
ic_cols_ub = find(cellfun(@(x) contains(x, 'ub_ic_x') || contains(x, 'ub_ic_y'), cols));
g_ic_ub = tbl{1, ic_cols_ub};

g_obs = {};
g_obs_lb = {};
g_obs_ub = {};
tmp = find(cellfun(@(x) contains(x, 'obs_0') && ~contains(x, 'lb_') && ~contains(x, 'ub_')  && ~contains(x, '_x') && ~contains(x, '_y') && ~contains(x, '_r'), cols));
i = 0;
while ~isempty(tmp)
    tmp_lb = find(cellfun(@(x) contains(x, ['lb_obs_' num2str(i)]), cols));
    tmp_ub = find(cellfun(@(x) contains(x, ['ub_obs_' num2str(i)]), cols));
    g_obs = [g_obs tbl{:, tmp}];
    g_obs_lb = [g_obs_lb tbl{1, tmp_lb}];
    g_obs_ub = [g_obs_ub tbl{1, tmp_ub}];
    i = i + 1;
    tmp = find(cellfun(@(x) contains(x, ['obs_' num2str(i)]) && ~contains(x, 'lb_') && ~contains(x, 'ub_')  && ~contains(x, '_x') && ~contains(x, '_y') && ~contains(x, '_r'), cols));
end

% Compute violations
dyn_viol = max(max(g_dyn - g_dyn_ub, 0), max(g_dyn_lb - g_dyn, 0));
obs_viol = cell(size(g_obs));
for i = 1:size(g_obs, 2)
    obs_viol{i} = max(max(g_obs{i} - g_obs_ub{i}, 0), max(g_obs_lb{i} - g_obs{i}, 0));
end
tube_viol = max(max(g_tube - g_tube_ub, 0), max(g_tube_lb - g_tube, 0));
ic_viol = max(max(g_ic - g_ic_ub, 0), max(g_ic_lb - g_ic, 0));

obs_x = tbl{1, cellfun(@(x) contains(x, 'obs_') && contains(x, '_x'), cols)};
obs_y = tbl{1, cellfun(@(x) contains(x, 'obs_') && contains(x, '_y'), cols)};
obs_r = tbl{1, cellfun(@(x) contains(x, 'obs_') && contains(x, '_r'), cols)};
z0 = tbl{1, cellfun(@(x) contains(x, 'z_ic'), cols)};
zf = tbl{1, cellfun(@(x) contains(x, 'z_g'), cols)};

%% Visualize
fh = figure(1);
clf;
subplot(2,2,1)
hold on;
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
for k = 1:size(w, 2)
    r = max(w(1, k), 0);
    tube{k} = rectangle('Position', [z(1, k, 1)-r, z(1, k, 2)-r, 2*r, 2*r], ...
        'Curvature', [1, 1], ...
        'EdgeColor', 'b', ...
        'LineWidth', 1);
end
z_line = plot(z(1, :, 1), z(1, :, 2), 'k', LineWidth=2);
xlabel('x')
ylabel('y')
axis equal
subplot(2,2,2)
hold on
gv_dyn_line = plot(dyn_viol(1, :), '.-', DisplayName="Dynamics");
gv_obs_lines = cell(size(obs_viol));
for i = 1:size(obs_viol, 2)
    gv_obs_line{i} = plot(obs_viol{i}(1, :), '.-', DisplayName=['Obstacle ' num2str(i)]);
end
gv_tub_line = plot(tube_viol(1, :), '.-', DisplayName="Tube");
gv_ic_line = plot(ic_viol(1, :), '.-', DisplayName="IC");
legend()
title("Constraint Violation")
subplot(2,2,3)
hold on
state_x_line = plot(squeeze(z(1, :, 1)));
state_y_line = plot(squeeze(z(1, :, 2)));
legend('x', 'y')
title("State")
subplot(2,2,4)
hold on;
input_x_line = plot(squeeze(v(1, :, 1)));
input_y_line = plot(squeeze(v(1, :, 2)));
tube_line = plot(w(1, :));
legend('$v_x$', '$v_y$', '$w$')
title("Input")
subplot(2, 2, 1)

%% Animate
for it = 1:size(iters, 1)
    for k = 1:size(w, 2)
        r = max(w(it, k), 0);
        tube{k}.Position = [z(it, k, 1)-r, z(it, k, 2)-r, 2*r, 2*r];
    end
    z_line.XData = z(it, :, 1);
    z_line.YData = z(it, :, 2);
    title(["Iteration: " iters(it)])
    gv_dyn_line.YData = dyn_viol(it, :);
    for i = 1:size(obs_viol, 2)
        gv_obs_line{i}.YData = obs_viol{i}(it, :);
    end

    gv_tub_line.YData = tube_viol(it, :);
    gv_ic_line.YData = ic_viol(it, :);
    state_x_line.YData = squeeze(z(it, :, 1));
    state_y_line.YData = squeeze(z(it, :, 2));
    input_x_line.YData = squeeze(v(it, :, 1));
    input_y_line.YData = squeeze(v(it, :, 2));
    tube_line.YData = w(it, :);
    drawnow
    pause
end
