function wb = subgradient(x, y, u, maxiter)
    % This function is to find the optimal weights w and bias b.
    dim = size(x, 2);
    w = zeros(dim, 1); % initial weights
    b=0; % initial bias
    N = size(x, 1);
    x = zscore(x);% normalization

    wb = [w; b];
    x = [x, ones(N, 1)];
    C = 0.01; % trade-off coeff.
    %% Training (Using Stochastic Gradient Descend-SGD):
    for iter1 = 1:maxiter
        S = randperm(N); % Shuffle the set
        for iter2 = S
            % sub-gradient: iter for each data point
            if y(iter2) * wb' * x(iter2,:)' <= 1
                wb = (1 - u) * [w; 0] + u * C * N * y(iter2) * x(iter2, :)'; 
                w = wb(1:end - 1);
            else
                w = (1 - u) * w;
                wb = [w; wb(end)];
            end
        end
    end
end