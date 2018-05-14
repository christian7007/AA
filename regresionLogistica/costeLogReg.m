function [J_cost, grad] = costeLogReg(theta, X, y, lambda)
	
	m = length(y);
	J_cost = 1/m * (-y'*log(sigmoide(X*theta)) - (1.-y)'*log(1.-sigmoide(X*theta)));
	J_cost = J_cost .+ (lambda/(2*m)) * sum((theta.^2));
	grad = 1/m * (sigmoide(X*theta) - y)' * X;
	
end
