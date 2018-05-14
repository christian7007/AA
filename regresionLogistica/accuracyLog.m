function accuracy = accuracyLog(theta, X, y)
	valor = round(sigmoide(X*theta));
	accuracy = 1 - sum(abs(valor - y)) / length(y);
end
