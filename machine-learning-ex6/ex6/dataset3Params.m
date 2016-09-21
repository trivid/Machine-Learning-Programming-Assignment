function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.1, 0.3, 1, 3, 10, 30];
error_val = zeros(length(C_list), length(sigma_list));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i=1:length(C_list)
    for j=1:length(sigma_list)
        C_i = C_list(i);
        sigma_j = sigma_list(j);
        model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
        visualizeBoundary(X, y, model);
        predictions = svmPredict(model, Xval);
        error_val(i, j) = mean(double(predictions ~= yval));        
    end    
end

[min_of_col, c] = min(error_val);
[min_of_row, r] = min(min_of_col);

C = C_list(c(r));
sigma = sigma_list(r);
% =========================================================================

end
