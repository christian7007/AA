function [J_cost, grad] = costeLog(theta, X, y)
	m = length(y);
	J_cost = 1/m * (-y'*log(sigmoide(X*theta)) - (1.-y)'*log(1.-sigmoide(X*theta)));
	grad = 1/m * (sigmoide(X*theta) - y)' * X;
end
