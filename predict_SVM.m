function Y_hat = predict_SVM(wb,Xt)
    Nt = size(Xt, 1);
    Xt = [Xt, ones(Nt, 1)];
    Xt = zscore(Xt);
    Y_hat = sign(Xt * wb);
end
