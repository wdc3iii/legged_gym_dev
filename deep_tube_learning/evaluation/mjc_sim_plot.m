mjc_data = readtable('data.csv');

% mjc_inds = mjc_data.t <= 3.5 & mjc_data.t >= 3;
% mjc_data = mjc_data(mjc_inds, :);
t_mjc = mjc_data.t;
pos_mjc = [mjc_data.x, mjc_data.y, mjc_data.z];
vel_mjc = [mjc_data.vx, mjc_data.vy, mjc_data.vz];
foot_mjc = mjc_data.legpos;
vfoot_mjc = mjc_data.legvel;
ang_vel_mjc = [mjc_data.w_1, mjc_data.w_2, mjc_data.w_3];
quat_mjc = [mjc_data.q_x, mjc_data.q_y, mjc_data.q_z, mjc_data.q_w];
quat_mjc(quat_mjc(:, 4) < 0, :) = quat_mjc(quat_mjc(:, 4) < 0, :) * -1;
quat_des_mjc = [mjc_data.qd_x, mjc_data.qd_y, mjc_data.qd_z, mjc_data.qd_w];
tau_foot_mjc = mjc_data.tau_foot;
wheel_vel_mjc = [mjc_data.wheel_vel1, mjc_data.wheel_vel2, mjc_data.wheel_vel3];
rom_mjc = [mjc_data.rom_x, mjc_data.rom_y, mjc_data.rom_vx, mjc_data.rom_vy];
torque_mjc = [mjc_data.tau1, mjc_data.tau2, mjc_data.tau3];

make_contact_mjc = t_mjc(diff(mjc_data.contact) > 0.5);
break_contact_mjc = t_mjc(diff(mjc_data.contact) < -0.5);

figure(3)
clf
hold on
plot(t_mjc, quat_mjc)
set(gca,'ColorOrderIndex',1)
plot(t_mjc, quat_des_mjc, '--')
ylabel('quat')
legend('mjc x', 'mjc y', 'mjc z', 'mjc w', 'des x', 'des y', 'des z', 'des w', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(4)
clf
plot(t_mjc, quat_mjc)
ylabel('quat')
legend('mjc x', 'mjc y', 'mjc z', 'mjc w', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(5)
clf
plot(t_mjc, quat_des_mjc)
ylabel('quat')
legend('des x', 'des y', 'des z', 'des w', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(6)
clf
ylabel('Angular velocity')
plot(t_mjc, ang_vel_mjc)
legend('mjc w_x', 'mjc w_y', 'mjc w_z', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(7)
clf
hold on
plot(t_mjc, pos_mjc)
ylabel('position')
legend('mjc x', 'mjc y', 'mjc z', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(8)
clf
plot(t_mjc, vel_mjc)
ylabel("velocity")
legend('mjc v_x', 'mjc v_y', 'mjc v_z', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(9)
clf
subplot(3, 1, 1)
plot(t_mjc, foot_mjc)
ylabel('Foot Pos')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')
subplot(3, 1, 2)
plot(t_mjc, vfoot_mjc)
ylabel('Foot Vel')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')
subplot(3, 1, 3)
plot(t_mjc, tau_foot_mjc)
ylabel('Foot Force')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(10)
clf
plot(t_mjc, wheel_vel_mjc)
ylabel('Wheel Vel')
legend('mjc wheel 1', 'mjc wheel 2', 'mjc wheel 3', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(11)
clf
plot(t_mjc, torque_mjc)
ylabel('Torque')
legend('mjc wheel 1', 'mjc wheel 2', 'mjc wheel 3', AutoUpdate='off')
xline(make_contact_mjc, 'r')
xline(break_contact_mjc, 'k')

figure(12)
clf
plot(t_mjc, rom_mjc)
ylabel('RoM')
legend('x', 'y', 'vx', 'vy', AutoUpdate='off')
