function [theta, J_history,theta_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = [theta' ; zeros(num_iters,2)];

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	sum=[0;0];
	for i = 1:m
		h=theta'*X(i,:)';
		sum=sum+(y(i)-h)*X(i,:)';
	end;
	theta = theta + alpha/m*sum;
	
	#fprintf('iter: %f\n',iter);
	#fprintf('Theta computed from gradient descent: (%f,%f)\n',theta(1),theta(2));
	#fprintf('Cost J(theta): %f\n',computeCost(X,y,theta));
	#fprintf('Gradient DJ(theta): (%f,%f)\n',sum(1)/m,sum(2)/m);
	#disp('----------------');
	

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	theta_history(iter+1,:) = [theta];

end

end
