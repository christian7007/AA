function accu = accuracyNN(Theta1, Theta2, X, y)

	a1 = [ones(rows(X), 1) X];
    z2 = Theta1 * a1';
    a2 = sigmoide(z2);
    a2 = [ones(1, columns(a2)); a2];
    z3 = Theta2 * a2;
    hipotesis = sigmoide(z3);

	[max field] = max(hipotesis);
	mask = (field' == y);

	good = length(find(mask == 1));
	accu = (good/length(y)) * 100;

end