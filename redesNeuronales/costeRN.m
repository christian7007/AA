function [J grad] = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)

    Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
    Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));

    m = length(y);

    yiden = eye(num_etiquetas);
    ycod = yiden(y, :);

    % propagacion hacia delante
    a1 = [ones(rows(X), 1) X];
    z2 = Theta1 * a1';
    a2 = sigmoide(z2);
    a2 = [ones(1, columns(a2)); a2];
    z3 = Theta2 * a2;
    hipotesis = sigmoide(z3);

    % coste
    J = 1/m * (sum(sum(((-ycod) .* log(hipotesis)') - ((1 - ycod) .* log(1 - hipotesis)'))));
    J = J + (lambda/(2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

    sigmoide_capa_3 = hipotesis' - ycod;
    sigmoide_capa_2 = (Theta2' * sigmoide_capa_3') .* derSigmoide([ones(1, columns(z2)); z2]);

    delta_1 = sigmoide_capa_2(2:end, :) * a1;
    delta_2 = sigmoide_capa_3' * a2';

    gradiente_1 = ((1/m) * delta_1);
    gradiente_2 = ((1/m) * delta_2);

    gradiente_1(:, 2:end) += (lambda/m) * Theta1(:, 2:end);
    gradiente_2(:, 2:end) += (lambda/m) * Theta2(:, 2:end);

    grad = [gradiente_1(:); gradiente_2(:)];
end
