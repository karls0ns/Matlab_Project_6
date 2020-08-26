function [SAE, MAE, SSE, MSE, RMSE, R2] = linreg_loocv(x, y, degree)
% funkcija apr��ina k��du krit�riju v�rt�bas ar LOOCV metodi
    n = size(y,1);
    yHat = zeros(n,1);
    for i = 1 : n
        % apm�c�bas kopas izveido�ana
        xTrain = x;
        xTrain(i,:) = []; % �eit notiek i-t� piem�ra dz��ana no apm�c�bas kopas
        yTrain = y;
        yTrain(i) = []; % �eit notiek i-t� piem�ra dz��ana no apm�c�bas kopas
        % parametru apr��in��ana
        a = linreg(xTrain, yTrain, degree);
        
        % valid��anas kopas izveido�ana no viena pa�a i-t� piem�ra
        xValid = x(i,:);
        %yValid nevajag, jo izmantosim evaluate funkciju ar vis�m prognoz�m vienlaic�gi
        
        % prognozes veik�ana i-taj� piem�r�, kas netika iek�auts apm�c�bas kop�
        yHat(i) = linreg_predict(xValid, a, degree);
    end
    [SAE, MAE, SSE, MSE, RMSE, R2] = evaluate(y, yHat);
end
