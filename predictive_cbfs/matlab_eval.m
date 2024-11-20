clear; clc;

i = 0;
skip = 100;
% dn = 'double_single_int_hy1xxooi';  % baseline
% dn = 'double_single_int_7fm4dsfd';  % ?
% dn = 'double_single_int_ylfgdlks';  % check loss alpha = 0.9
% dn = 'double_single_int_9zyu99yc';   % Short horizon
% dn = 'double_single_int_uv3ck9sh';   % const step size
% dn = 'double_single_int_hhlxbg8m';   % longer run
dn = 'double_single_easy_int_yimjsolw';   % bigger set

quit = false;
while ~quit
    disp(i)
    load(['predictive_cbfs/' dn '/learning_iterates_' num2str(i) '.mat'])

    figure(1)
    subplot(2,2,1)
    scatter(x(1:skip:end, 1), x(1:skip:end, 2), 10, delta(1:skip:end))
    hold on
    xb = xlim;
    yb = ylim;
    xplt = linspace(xb(1), xb(2));
    yplt = - 5 * xplt;
    inds = yplt < yb(2) & yplt > yb(1);
    plot(xplt(inds), yplt(inds), 'r')
    hold off
    colorbar
    xlabel('x')
    ylabel('dot x')
    title(['delta ' num2str(i)])

    subplot(2,2,2)
    scatter(x(1:skip:end, 1), x(1:skip:end, 2), 10, viol(1:skip:end))
    hold on
    xb = xlim;
    yb = ylim;
    xplt = linspace(xb(1), xb(2));
    yplt = - 5 * xplt;
    inds = yplt < yb(2) & yplt > yb(1);
    plot(xplt(inds), yplt(inds), 'r')
    hold off
    colorbar
    xlabel('x')
    ylabel('dot x')
    title(['Violation ' num2str(i)])

    subplot(2,2,3)
    scatter(x(1:skip:end, 1), x(1:skip:end, 2), 10, delta_targ(1:skip:end))
    hold on
    xb = xlim;
    yb = ylim;
    xplt = linspace(xb(1), xb(2));
    yplt = - 5 * xplt;
    inds = yplt < yb(2) & yplt > yb(1);
    plot(xplt(inds), yplt(inds), 'r')
    hold off
    colorbar
    xlabel('x')
    ylabel('dot x')
    title(['delta target ' num2str(i)])

    subplot(2,2,4)
    histogram(viol(viol > 0))
    xlabel('Violation')
    ylabel('Count')
    title('Violation Histogram')

    % key = input("", "s");
    % 
    % if numel(key) == 0
    %     i = i + 1;
    %     if ~exist(['predictive_cbfs/' dn '/learning_iterates_' num2str(i) '.mat'], "file")
    %         disp("end of files reached...")
    %     end
    % elseif key == 'q'
    %     disp("quitting")
    %     quit = true;
    % else
    %     i = max(i - 1, 0);
    % end
    di = 1;
    % if i < 50
    %     di = 3;
    % end
    i = i + di;
    drawnow
end