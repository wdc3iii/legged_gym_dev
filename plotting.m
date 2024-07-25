load('/home/wcompton/Repos/legged_gym_dev/legged_gym/scripts/play_data.mat')

t = [0:size(cmd, 1)-1] * 0.02;
figure(1)
clf
plot(t, cmd')
ylabel('command')

figure(2)
clf
plot(t, action')
ylabel('action')

figure(3)
clf
plot(t, (action ./ vecnorm(action, 2, 2))')
ylabel('normalized action')
legend('w_{des}', 'x_{des}', 'y_{des}', 'z_{des}')

figure(4)
clf
plot(t, quat')
ylabel('quat')
legend('x', 'y', 'z', 'w')

figure(5)
clf
plot(t, omega')
ylabel('Angular velocity')
legend('w_x', 'w_y', 'w_z')

figure(6)
clf
plot(t, pos')
ylabel('position')
legend('x', 'y', 'z')

figure(7)
clf
plot(t, vel')
ylabel("velocity")
legend('v_x', 'v_y', 'v_z')

figure(8)
clf
hold on
plot(t, dof(:, 1)')
plot(t, ddof(:, 1)')
ylabel('Foot')
legend('pos', 'vel')

figure(9)
clf
plot(t, ddof(:, 2:4)')
ylabel('Wheel Vel')
legend('wheel 1', 'wheel 2', 'wheel 3')


figure(10)
clf
plot(t, torque(:, 2:4)')
ylabel('Torque')
legend('wheel 1', 'wheel 2', 'wheel 3')