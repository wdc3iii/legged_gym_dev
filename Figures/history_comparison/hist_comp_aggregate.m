clear; clc;

set(groot, 'DefaultAxesFontSize', 17);  % Set default font size for axes labels and ticks
set(groot, 'DefaultTextFontSize', 17);  % Set default font size for text objects
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');  % Set interpreter for axis tick labels
set(groot, 'DefaultTextInterpreter', 'latex');  % Set interpreter for text objects (e.g., titles, labels)
set(groot, 'DefaultLegendInterpreter', 'latex')
set(groot, 'DefaultFigureRenderer', 'painters');
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'DefaultLineMarkerSize', 15)
H = [1,5,10,15,20,25,30,35,40,45,50];

%% Plot Mean Error When Correct vs. H
% Load in data
% rbot = 'hopper';
% rbot = 'double';
rbot = 'hopper_sluggish';

rec_data = readtable(['data/' rbot '_rec_data.csv']);
os_data = readtable(['data/' rbot '_os_data.csv']);

% filter variable names
col_rec = rec_data.Properties.VariableNames;
filt_col_rec = col_rec(~endsWith(col_rec, {'Step', '1', '2'}));
rec_data = rec_data(:, filt_col_rec);
col_os = os_data.Properties.VariableNames;
filt_col_os = col_os(~endsWith(col_os, {'Step', '1', '2'}));
os_data = os_data(:, filt_col_os);

% Plot best mean err
rec_data = rec_data(90:end, :);
best_rec_err = min(rec_data{:, :}, [], 1);
os_data = os_data(90:end, :);
best_os_err = min(os_data{:, :}, [], 1);

fh = figure(2);
fh.Position = [266 1232 774 566];
clf;
hold on
plot(H, flip(best_rec_err), '.-')
plot(H, flip(best_os_err), '.-')
xlabel("H")
legend("Recursive", "Oneshot")
ylabel("Mean Error when Correct")