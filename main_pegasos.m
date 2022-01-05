clear all
load("linear_svm.mat");
%% load training data and initialization:
u = 1e-6; % learning rate for stochastic gradient descent (SGD)
maxiter = 200; 

%% Training: 
wb = subgradient(X_train, labels_train, u, maxiter);

%% Test:
Y_hat = predict_SVM(wb, X_test);

%% Results:
test_accuracy = sum(Y_hat == labels_test) / numel(labels_test) * 100
% plotconfusion(categorical(Yt),categorical(Y_hat'))

% plot the given datasets
w = wb(1:end - 1);
b = wb(end);
visualize(zscore(X_train), labels_train, w, b, 'soft margin results from Pegasos on training data');
visualize(zscore(X_test), labels_test, w, b, 'soft margin results from Pegasos on testing data');
