clear; clc;

N = 25;
H_max = 25;
nz = 1;
ne = 1;
nw = 1;
nv = 1;

figure(1)
clf;
t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(1);
j = 0;
for H = [25, 12, 1]
    j = j + 1;

    dense_rec_e = zeros(N, H_max * ne);
    dense_rec_w = zeros(N, N * nw);
    dense_rec_z = zeros(N, (H_max + N) * nz);
    dense_rec_v = zeros(N, (H_max + N-1) * nv);
    dense_os_e = zeros(N, H_max * ne);
    dense_os_w = zeros(N, N * nw);
    dense_os_z = zeros(N, (H_max + N) * nz);
    dense_os_v = zeros(N, (H_max + N-1) * nv);

    % set os connectivity
    dense_os_e(:, (H_max - H+1) * ne:end) = 1;
    dense_os_z(:, (H_max - H+1) * nz:end) = 1;
    dense_os_v(:, (H_max - H+1) * nv:end) = 1;
    dense_os_w(logical(eye(size(dense_os_w)))) = 1;
    dense_os = [dense_os_e, dense_os_w, dense_os_z, dense_os_v];

    % set rec connectivity
    for i = 1:N
        dense_rec_e(i, (H_max - H + i) * ne:end) = 1;
        dense_rec_w(i, max(1, (i-H) * nw):i * nw) = 1;
        dense_rec_z(i, (H_max - H + i) * nz:H_max + (i-1)*nz) = 1;
        dense_rec_v(i, (H_max - H + i) * nv:H_max + (i-1)*nv) = 1;
    end
    dense_rec = [dense_rec_e, dense_rec_w, dense_rec_z, dense_rec_v];

    nexttile(j);
    imagesc(dense_os);
    colormap([1 1 1; 0.5 0.8 1]); % Custom colormap: [white; light blue]
    hold on
    xline(0.5+[H_max * ne, H_max * ne + N * nw, H_max * ne + N * nw + (H_max + N) * nz], 'k', LineWidth=2)
    xline(0.5+[H_max * ne + N * nw + H_max * nz, H_max * ne + N * nw + (H_max + N) * nz + H_max * nv], 'k', LineWidth=1)
    axis xy;                      % Flip y-axis to have 1 at the top and N at the bottom
    colorbar off;                 % Turn off colorbar if not needed
    axis equal
    axis tight
    title(['Oneshot, H=' num2str(H)])

    j = j + 1;
    nexttile(j)
    imagesc(dense_rec);
    colormap([1 1 1; 0.5 0.8 1]); % Custom colormap: [white; light blue]
    hold on
    xline(0.5+[H_max * ne, H_max * ne + N * nw, H_max * ne + N * nw + (H_max + N) * nz], 'k', LineWidth=2)
    xline(0.5+[H_max * ne + N * nw + H_max * nz, H_max * ne + N * nw + (H_max + N) * nz + H_max * nv], 'k', LineWidth=1)
    axis xy;                      % Flip y-axis to have 1 at the top and N at the bottom
    colorbar off;                 % Turn off colorbar if not needed
    axis equal
    axis tight
    title(['Recursive, H=' num2str(H)])
end