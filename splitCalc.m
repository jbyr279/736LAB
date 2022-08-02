% Clear workspace
clear
clc

% Number of trials in data
trials = 36;

% Load and shorten data
load("waveform.mat");
data = data(1:1800000);

% FFT signal
Fsig_ = fft(data) / length(data);
aFsig_ = abs(Fsig_);

% Split signal into trials
trial_data = cell(1,trials);
for i = 0:trials-1
    trial_data{i+1} = data(i*50000 + 1:(i+1)*50000)';
end

% Store power spectrum results in cell array 
alpha = zeros(1,36);
beta = zeros(1,36);
for i = 1:length(trial_data)
    [pow_a, pow_b] = calculatePower(trial_data{i}, aFsig_(1));
    alpha(i) = pow_a;
    beta(i) = pow_b;
    
    % Create group vector
    if mod(i,2) == 1
        status(i) = "CLOSE";
    else
        status(i) = "OPEN";
    end

end

% Set up figure
figure; hold on

% Seperate data
plot(alpha(1:2:end), beta(1:2:end), "o", "Color", [1 0 0]);
plot(alpha(2:2:end), beta(2:2:end), "o", "Color", [0 0 1]);

title("Alpha Band VS Beta Band Power")
xlabel("Alpha Band Power (dB)")
ylabel("Beta Band Power (dB)")

xlim([-18 0]);
ylim([-30 0]);

labels = ["Eyes Closed" , "Eyes Open"];

legend(labels, "Location","northwest");

% Sort classifier hold-in and hold-out trials
power_band = [alpha', beta'];

% Result array
res = [];

% Classify data and test algo
for i = -1:2:trials-3 
    train_data = power_band;
    trial_status = status;

    sample_inds = [(trials - i - 2) (trials - i - 1)];
    sample_data = power_band(sample_inds, :);

    train_data(sample_inds, :) = [];
    trial_status(sample_inds) = [];
    
    % Get decision border 
    [C,err,posterior,logp,coeff] = classify(sample_data, train_data, trial_status);
    res = [res; C];

    K = coeff(1,2).const;
    L = coeff(1,2).linear;
    
    % Set up function
    f = @(alpha, beta) K + L(1)*alpha + L(2)*beta;

end

% Plot decision boundary
figure; hold on
h3 = fimplicit(f);
h3.Color = 'm';
h3.LineWidth = 2;
plot(sample_data(:, 1), sample_data(:,2), "o", "Color", [1 0 0]);
plot(train_data(:, 1), train_data(:,2), "o", "Color", [0 0 1]);

xlim([-18 0]);
ylim([-30 0]);

title("Alpha Band VS Beta Band Power (Decision Boundary After Classification)")
xlabel("Alpha Band Power (dB)")
ylabel("Beta Band Power (dB)")

labels = ["Decision Boundary", "Hold Out" , "Hold In"];

legend(labels, "Location","northwest");

hold off

res = [res, status'];

% Set up success rate
for i = 1:trials
    if (res(i,1) == res(i,2)) 
        res(i,3) = 1;
    else
        res(i,3) = 0;
    end
end

% Calc success rate
pct_succ = (1-length(find(res(:,3) == "0"))/trials) * 100;















clearvars -except res pct_succ
