function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n=length(theta);
sumth_sq = 0;
for i=2:n
	sumth_sq+=theta(i)^2;
end;

H_th = zeros(m);
for i=1:m
	% Calculation of hypothesis value h_theta(x)
	% h_th = sigmoid(theta'*X(i,:)');
	h_th = sigmoid(X(i,:)*theta);
	H_th(i)=h_th;
	
	% Summation of cost
	if y(i) == 1
		J -= log(h_th);
	elseif y(i) == 0
		J-= log(1-h_th);
	end
end

for j=1:n
	for i=1:m
		% Summation of gradient entries
		%h_th = sigmoid(X(i,:)*theta);
		h_th = H_th(i);
		grad(j) += (h_th - y(i))*X(i,j);
	end
	if j!=1
		grad(j) +=  lambda * theta(j);
	end
end

J/=m;
J+= sumth_sq * lambda/(2*m);
grad./=m;




% =============================================================

end
