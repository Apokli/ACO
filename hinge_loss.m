function loss = hinge_loss(real, pred)
% hinge_loss - calculates the hinge loss of the prediction.
% Inputs:
%   real - given classes
%   pred - predicted classes
    tmp_vec = zeros(size(real, 1), 1);
    tmp_loss = 1 - real .* pred;
    loss = sum(max(tmp_vec, tmp_loss));
end

