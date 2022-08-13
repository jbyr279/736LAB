% Clear workspace
clear
clc

%% Wrangle Data
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
        yn(i) = -1;
    else
        status(i) = "OPEN";
        yn(i) = 1;
    end

end

figure; hold on

plot(alpha(1:2:end), beta(1:2:end), "or"); 
plot(alpha(2:2:end), beta(2:2:end), "ob");

axis([40 60 30 50])

title("Two-Dimensional Feature Space")
xlabel('Beta Band Power (dB)')
ylabel('Alpha Band Power (dB)')

labels = ["Closed" , "Open"];
legend(labels, "Location","best");

%% Fisher's linear discriminant
% Sort classifier hold-in and hold-out trials
power_band = [alpha', beta'];

% Result array
res = [];
pred_svm = [];

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
    

    svm = fitcsvm(train_data, trial_status);
    label = predict(svm, sample_data);

    pred_svm = [pred_svm; label];

end

%% Plot results
% Plot decision boundary
figure; hold on
h3 = fimplicit(f);
h3.Color = 'k';
h3.LineWidth = 1;

plot(alpha(1:2:end), beta(1:2:end), "or"); 
plot(alpha(2:2:end), beta(2:2:end), "ob"); 
plot(sample_data(:, 1), sample_data(:,2), "xk", 'MarkerSize', 12); 

axis([40 60 30 50])

title("Fisher's Linear Discriminant")
xlabel("Alpha Band Power (dB)")
ylabel("Beta Band Power (dB)")

labels = ["Decision Boundary", "Closed" , "Open", "Hold-Out"];

legend(labels, "Location","best");

hold off

pred_svm = [pred_svm, status'];
res = [res, status'];

% Set up success rate
for i = 1:trials
    if (res(i,1) == res(i,2)) 
        res(i,3) = 1;
    else
        res(i,3) = 0;
    end

    if (pred_svm(i,1) == pred_svm(i,2)) 
        pred_svm(i,3) = 1;
    else
        pred_svm(i,3) = 0;
    end
end

%% Plot trial data for feature analysis
close all
figure;
t_sec = 1:50000;
plot(t_sec, trial_data{13})

title("EEG Raw Signal - Eyes Closed")
xlabel("Time (10^{-4} s)")
ylabel("Measured Signal (V)")

ylim([-0.04, 0.04])

figure;
t_sec = 1:50000;
plot(t_sec, trial_data{14})

title("EEG Raw Signal - Eyes Open")
xlabel("Time (10^{-4} s)")
ylabel("Measured Signal (V)")

ylim([-0.04, 0.04])

%% Post proc items for subsequent tasks
% Calc success rate
pct_succ = (1-length(find(res(:,3) == "0"))/trials) * 100;

power_band = [power_band, yn'];

%% T-test 

FLD = cellfun(@str2num, res_fld(:,3));
SVM = cellfun(@str2num, res_svm(:,3));

[H,p,CI] = ttest2(FLD,SVM);

