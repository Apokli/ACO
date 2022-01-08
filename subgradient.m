function w = subgradient(x, y, u, maxIter)
% This function is to find the optimal weights w.
% set bias b to be zero.
    dim = size(x, 2);
    w = zeros(dim, 1); % initial weights
    N = size(x, 1);
    x=  zscore(x);% standardization
    C = 0.01; % trade-off coeff.
    % C=15
    %% Training (Using Pegasos):
    for iter1 = 1 : maxIter
        % pick one data point in one iteration until MaxIter
        iter2 = randi(N);
        if y(iter2) * w' * x(iter2, :)' <= 1
           w = (1 - u) * w + u * C * N * y(iter2) * x(iter2, :)'; 
        else
           w = (1 - u) * w;
        end
    end 
    %% Training (Using mini-batch Pegasos):
    a = 1; % if a=0.01, is the same as Pegasos
    k = a * N;
    for iter1 = 1:maxIter
        S = randperm(N); % Shuffle the set
        S = S([1 : k]);
        xx = x(S, :); % new data set
        yy = y(S);
        temp = zeros(dim, length(S));
        for iter2 = 1:length(S)
            % sub-gradient: iter for each data point in the subset for every
            % iteration
            if yy(iter2) * w' * xx(iter2, :)' <= 1
               temp(:, iter2) = u * C * N * yy(iter2) * xx(iter2, :)';     
            else
               temp(:, iter2) = zeros(dim, 1);
            end        
        end
        w = (1 - u) * w + mean(temp, 2);
    end
end