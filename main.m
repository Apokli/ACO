clear all
close all

load("linear_svm.mat");

% normalize dataset
X_train = zscore(X_train);

%% solve the softened (soft margin) problem via CVX

lambda = 1000; % softenning coefficient
cvx_begin
    variable w(2) 
    variable b
    minimize(lambda * w.'* w / 2 + hinge_loss(labels_train, zscore([X_train, ones(size(X_train , 1), 1)]) * [w;b])) 
cvx_end
 
% calculate test accuracy
wb = [w;b];
Y_hat = predict_SVM(wb,X_test);
test_accuracy = sum(Y_hat == labels_test) / numel(labels_test) * 100

% plot the results
visualize(zscore(X_train), labels_train, w,b, 'soft margin results from CVX on training data');
visualize(zscore(X_test), labels_test, w, b, 'soft margin results from CVX on testing data');

%% solve the softened (soft margin) problem via Pegasos
u = 1e-5; % learning rate for stochastic gradient descent (SGD), set a small number
maxIter = 1e2; 

tStart = cputime;
w = subgradient(X_train, labels_train, u, maxIter);
tEnd = cputime - tStart

%set b directly to zero
wb = [w; 0];
Y_hat = predict_SVM(wb, X_test);

format long
test_accuracy = sum(Y_hat == labels_test) / numel(labels_test) * 100

% plot the given datasets
w = wb(1:end-1);
b = wb(end);

visualize(zscore(X_train), labels_train, w, b, 'soft margin results from Pegasos on training data');
visualize(zscore(X_test), labels_test, w, b, 'soft margin results from Pegasos on testing data');