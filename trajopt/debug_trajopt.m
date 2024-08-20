tbl = readtable("debug_trajopt_results_L2_dense10.csv");

iters = tbl.iter;

z = tbl{:, 2:103};
zx = z(:, 1:2:end);
zy = z(:, 2:2:end);
z = cat(3, zx, zy);

v = tbl{:, 104:203};
vx = v(:, 1:2:end);
vy = v(:, 2:2:end);
v = cat(3, vx, vy);

w = tbl{:, 204:254};

g = tbl{:, 255:456};
glb = tbl{1, 457:658};
gub = tbl{1, 659:860};
xlb = tbl{1, 861:1113};
xub = tbl{1, 1114:end};

% Compute violations
g_violation = max(max(g - gub, 0), max(glb - g, 0));
idx = 1;
g_dynamics = [];
g_obstacle = [];
g_tube = [];

for i = 1:size(v, 2)
    g_dynamics = [g_dynamics idx idx+1];
    idx = idx + 2;
    g_obstacle = [g_obstacle idx];
    idx = idx + 1;
end

g_tube = [idx:idx + size(w, 2) - 2];
idx = idx + size(w, 2) - 1;
g_obstacle = [g_obstacle idx];
idx = idx + 1;
g_ic = [idx];

%% Visualize
fh = figure(1);
clf;
subplot(2,2,1)
hold on;
plot(0, 0, 'go')
plot(4, 3, 'rx')
obs_x = [2, 0];
obs_y = [1.5, 3];
obs_r = [1, 1];
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
    r = w(1, k);
    tube{k} = rectangle('Position', [z(1, k, 1)-r, z(1, k, 2)-r, 2*r, 2*r], ...
          'Curvature', [1, 1], ...
          'EdgeColor', 'b', ...
          'LineWidth', 1);
end
z_line = plot(z(1, :, 1), z(1, :, 2), 'k', LineWidth=2);
xlabel('x')
ylabel('y')
xlim([-1, 5])
ylim([-1, 5])
axis square
subplot(2,2,2)
hold on
gv_dyn_line = plot(g_violation(1, g_dynamics), '.-');
gv_obs_line = plot(g_violation(1, g_obstacle), '.-');
gv_tub_line = plot(g_violation(1, g_tube), '.-');
gv_ic_line = plot(g_violation(1, g_ic), '.-');
legend('Dynamics Viol', 'Obstacle Viol', 'Tube Dyn Viol', 'IC Viol')
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
legend('v_x', 'v_y')
title("Input")
subplot(2, 2, 1)

while (1)
    for it = 1:size(iters, 1)
        for k = 1:size(w, 2)
            r = w(it, k);
            tube{k}.Position = [z(it, k, 1)-r, z(it, k, 2)-r, 2*r, 2*r];
        end
        z_line.XData = z(it, :, 1);
        z_line.YData = z(it, :, 2);
        title(["Iteration: " it])
        gv_dyn_line.YData = g_violation(it, g_dynamics);
        gv_obs_line.YData = g_violation(it, g_obstacle);
        gv_tub_line.YData = g_violation(it, g_tube);
        gv_ic_line.YData = g_violation(it, g_ic);
        state_x_line.YData = squeeze(z(it, :, 1));
        state_y_line.YData = squeeze(z(it, :, 2));
        input_x_line.YData = squeeze(v(it, :, 1));
        input_y_line.YData = squeeze(v(it, :, 2));
        drawnow
    end
    pause(1)
end