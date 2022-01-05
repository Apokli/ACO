clear all
close all

load("linear_svm.mat");

% normalize dataset
X_train = zscore(X_train);

%% solve original (hard margin) problem
cvx_begin
    variable w(2) 
    variable b
    minimize(norm(w))
    subject to
        for i = 1:length(X_train)
            1 - labels_train(i) * (X_train(i, :) * w + b) <= 0
        end
cvx_end
 
% calculate test accuracy
wb = [w;b];
Y_hat = predict_SVM(wb,X_test);
test_accuracy = sum(Y_hat'==labels_test)/numel(labels_test) * 100

% plot the results
visualize(zscore(X_train), labels_train, w, b, 'hard margin results from CVX on training data');
visualize(zscore(X_test), labels_test, w, b, 'hard margin results from CVX on testing data');

%% solve the softened (soft margin) problem
lambda = 1000; % softenning coefficient
cvx_begin
    variable w(2) 
    variable b
    minimize(lambda * w.'* w / 2 + hinge_loss(labels_train, zscore([X_train, ones(size(X_train , 1), 1)]) * [w;b])) 
cvx_end
 
% calculate test accuracy
wb = [w;b];
Y_hat = predict_SVM(wb,X_test);
test_accuracy = sum(Y_hat'==labels_test)/numel(labels_test) * 100

% plot the results
visualize(zscore(X_train), labels_train, w,b, 'soft margin results from CVX on training data');
visualize(zscore(X_test), labels_test, w, b, 'soft margin results from CVX on testing data');


