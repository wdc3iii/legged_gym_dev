clear; clc;

tags = {...
    'sjiqi49f', ...
    '7p1zump7', ...
    'msj97p19', ...
    'ks1eg0xw', ...
    'rg5itafm', ...
    'j88i9kim', ...
    'mnfs3r5v', ...
    'r5xu847t', ...
    'l4wnnx72', ...
    'w8flp57h', ...
    'f9zr70ds' ...
};
H = [1,5,10,15,20,25,30,35,40,45,50];

plt_timing = zeros(size(tags, 2), 1);
for ii = 1:size(tags, 2)
nm = ['data/' tags{ii} '_rec.mat'];
load(nm);
t = mean(diff(timing(1:2))) / 1e9;
plt_timing(ii) = t;
end

figure(1);
clf
plot(H, plt_timing)
xlabel('H')
ylabel('Solve Time (s)')

plt_cv = zeros(size(tags, 2), 1);
for ii = 1:size(tags, 2)
nm = ['data/' tags{ii} '_rec_lim.mat'];
load(nm);
for k = 1:24
    cv = eval(['cv' num2str(k)]);
    plt_cv(ii) = plt_cv(ii) + sum(cv.Dynamics) + sum(cv.('Obstacle 0')) + sum(cv.('Obstacle 1')) + sum(cv.('Initial Condition')) + sum(cv.('Tube Dynamics'));
end
end
figure(2);
clf
plot(H, plt_cv)
xlabel('H')
ylabel('Net Constraint Violation')