function derivada = derSigmoide(Z)

    derivada = sigmoide(Z) .* (1 - sigmoide(Z));

end