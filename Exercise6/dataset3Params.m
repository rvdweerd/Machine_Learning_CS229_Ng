function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

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

results = zeros(64,3);
count=1;
it = [0.01,0.03,0.1,0.3,1,3,10,30];
for i=1:8
	C=it(i);
	for j=1:8
		sigma=it(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		p=svmPredict(model,Xval);
		err=mean(double(p~=yval));
		
		fprintf('sigma = %f\n', sigma);
		fprintf('C = %f\n', C);
		fprintf('err = %f\n', err);
		
		results(count,:)=[err,C,sigma];
		count+=1;
	end
end

[v,minIndex]=min(results(:,1));
C=results(minIndex,2);
sigma=results(minIndex,3);

%C = 0.3;
%sigma = 0.1;



% =========================================================================

end
