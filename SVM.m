% Get data from previous tasks
splitCalc;
DAT = power_band;

%% Set up SVM and classify
svm = fitcsvm(DAT(:,1:2), DAT(:,3), 'KernelFunction', 'linear');
crossval_svm = crossval(svm, 'kfold', 18);

f = kfoldLoss(crossval_svm);
pct_succ(2) = 1 - f;

DAT = power_band;

%% Plot SVM results
figure; hold on

plot(alpha(1:2:end), beta(1:2:end), "or"); 
plot(alpha(2:2:end), beta(2:2:end), "ob");

plot(svm.SupportVectors(:,1), svm.SupportVectors(:,2), 'xk', 'MarkerSize', 12)

axis([40 60 30 50])

title("Support Vector Machine")
xlabel('Beta Band Power (dB)')
ylabel('Alpha Band Power (dB)')

labels = ["Closed" , "Open", "Support Vectors"];
legend(labels, "Location","best");

%% Plot SVM margin visualised
% Decision regions
[X1,X2] = meshgrid(linspace(0,100,1000));
X1 = X1(:);
X2 = X2(:);
[label, score] = predict(svm, [X1 X2]);
R = reshape(label, [1000 1000]);
figure; hold on
imagesc(R); axis image off; colormap([0 0 1; 0 0 0]) 
title('Decision regions - blue and black')

% Margin
R = reshape(score(:,2), [1000 1000]);
figure; hold on
imagesc(double((R >= -1) & (R <= 1)), [0 1]); axis image off; colormap([0.7 0.7 0.7; 0 1 0])
title('Margin - Green')