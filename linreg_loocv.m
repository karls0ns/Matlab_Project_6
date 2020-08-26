function [SAE, MAE, SSE, MSE, RMSE, R2] = linreg_loocv(x, y, degree)
% funkcija aprçíina kïûdu kritçriju vçrtîbas ar LOOCV metodi
    n = size(y,1);
    yHat = zeros(n,1);
    for i = 1 : n
        % apmâcîbas kopas izveidoðana
        xTrain = x;
        xTrain(i,:) = []; % ðeit notiek i-tâ piemçra dzçðana no apmâcîbas kopas
        yTrain = y;
        yTrain(i) = []; % ðeit notiek i-tâ piemçra dzçðana no apmâcîbas kopas
        % parametru aprçíinâðana
        a = linreg(xTrain, yTrain, degree);
        
        % validçðanas kopas izveidoðana no viena paða i-tâ piemçra
        xValid = x(i,:);
        %yValid nevajag, jo izmantosim evaluate funkciju ar visâm prognozçm vienlaicîgi
        
        % prognozes veikðana i-tajâ piemçrâ, kas netika iekïauts apmâcîbas kopâ
        yHat(i) = linreg_predict(xValid, a, degree);
    end
    [SAE, MAE, SSE, MSE, RMSE, R2] = evaluate(y, yHat);
end
