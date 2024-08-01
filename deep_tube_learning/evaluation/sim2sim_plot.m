load('sim2sim.mat')
mjc_data = readtable('data.csv');
mjc_inds = mjc_data.t <= max(t);

mjc_data = mjc_data(mjc_inds, :);
t_mjc = mjc_data.t;
pos_mjc = [mjc_data.x, mjc_data.y, mjc_data.z];
vel_mjc = [mjc_data.vx, mjc_data.vy, mjc_data.vz];
foot_mjc = mjc_data.legpos;
vfoot_mjc = mjc_data.legvel;
ang_vel_mjc = [mjc_data.w_1, mjc_data.w_2, mjc_data.w_3];
quat_mjc = [mjc_data.q_x, mjc_data.q_y, mjc_data.q_z, mjc_data.q_w];
tau_foot_mjc = mjc_data.tau_foot;
wheel_vel_mjc = [mjc_data.wheel_vel1, mjc_data.wheel_vel2, mjc_data.wheel_vel3];
torque_mjc = [mjc_data.tau1, mjc_data.tau2, mjc_data.tau3];

make_contact = t(find(diff(contact) > 0.5));
break_contact = t(find(diff(contact) < -0.5));

make_contact_mjc = t_mjc(diff(mjc_data.contact) > 0.5);
break_contact_mjc = t_mjc(diff(mjc_data.contact) < -0.5);

figure(4)
clf
hold on
plot(t, [qx; qy; qz; qw]')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, quat_mjc, '--')
ylabel('quat')
legend('x', 'y', 'z', 'w', 'mjc x', 'mjc y', 'mjc z', 'mjc w', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')

figure(5)
clf
hold on
plot(t, [wx; wy; wz]')
ylabel('Angular velocity')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, ang_vel_mjc, '--')
legend('w_x', 'w_y', 'w_z', 'mjc w_x', 'mjc w_y', 'mjc w_z', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')

figure(6)
clf
hold on
plot(t, [x; y; z]')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, pos_mjc, '--')
ylabel('position')
legend('x', 'y', 'z', 'mjc x', 'mjc y', 'mjc z', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')

figure(7)
clf
hold on
plot(t, [vx; vy; vz]')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, vel_mjc, '--')
ylabel("velocity")
legend('v_x', 'v_y', 'v_z', 'mjc v_x', 'mjc v_y', 'mjc v_z', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')

figure(8)
clf
subplot(3, 1, 1)
hold on
plot(t, foot')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, foot_mjc, '--')
ylabel('Foot Pos')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')
subplot(3, 1, 2)
hold on
plot(t, vfoot)
set(gca,'ColorOrderIndex',1)
plot(t_mjc, vfoot_mjc, '--')
ylabel('Foot Vel')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')
subplot(3, 1, 3)
hold on
plot(t, tau_f)
set(gca,'ColorOrderIndex',1)
plot(t_mjc, tau_foot_mjc, '--')
ylabel('Foot Force')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')

figure(9)
clf
plot(t, [w1; w2; w3]')
ylabel('Wheel Pos')
legend('wheel 1', 'wheel 2', 'wheel 3', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')

figure(10)
clf
hold on
plot(t, [vw1; vw2; vw3]')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, wheel_vel_mjc, '--')
ylabel('Wheel Vel')
legend('wheel 1', 'wheel 2', 'wheel 3', 'mjc wheel 1', 'mjc wheel 2', 'mjc wheel 3', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')
xline(make_contact_mjc, 'r--')
xline(break_contact_mjc, 'k--')


figure(11)
clf
hold on
plot(t, [tau_w1; tau_w2; tau_w3]')
set(gca,'ColorOrderIndex',1)
plot(t_mjc, torque_mjc, '--')
ylabel('Torque')
legend('wheel 1', 'wheel 2', 'wheel 3', 'mjc wheel 1', 'mjc wheel 2', 'mjc wheel 3', AutoUpdate='off')
xline(make_contact, 'r')
xline(break_contact, 'k')