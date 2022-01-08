clear all
close all
%% load training data and initialization:
load("linear_svm.mat");
u = 1e-5; % learning rate for stochastic gradient descent (SGD), set a small number
maxIter = 1e2; 

%% Training: 
tStart = cputime;
w = subgradient(X_train, labels_train, u, maxIter);
tEnd = cputime - tStart

%% Test:
%set b directly to zero
wb = [w; 0];
Y_hat = predict_SVM(wb, X_test);

%% Accuracy:
format long
test_accuracy = sum(Y_hat == labels_test) / numel(labels_test) * 100

% plot the given datasets
w = wb(1:end-1);
b = wb(end);
% visualize(X_train, labels_train, w, b, 'training data');
% visualize(X_test, labels_test, w, b, 'testing data');
visualize(zscore(X_train), labels_train, w, b, 'training data');
visualize(zscore(X_test), labels_test, w, b, 'testing data');
