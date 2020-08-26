function yHat = linreg_predict(zq, a, degree)
% funkcija izmanto parametrus a, lai prognoz�tu yHat v�rt�bas dotaj�m zq faktoru ieejas v�rt�b�m
% arguments "degree" ir polinoma pak�pe
% arguments "a" ir apr��in�tie polinoma parametri
	m = size(zq,2);
    % noslinkosim - realiz�sim visu tikai viena faktora jebkuras pak�pes
    % polinoma regresijai un daudzfaktoru pirm�s pak�pes polinoma regresijai,
    % bet daudzfaktoru gad�jumam ar pak�pi >1 izvad�sim k��das pazi�ojumu
    if (degree <= 1) || (m <= 1)
        % pirmk�rt j�izveido x matrica t�pat k� linreg funkcij�
        x = ones(size(zq,1), 1);
        for i = 1 : degree
            x = [x zq .^ i];
        end
        % otrk�rt j�veic yHat apr��ins, izmantojot a un x
        %{
        % �eit ir gar� versija:
        yHat = zeros(size(x,1), 1);
        for i = 1 : size(x,1)
            for j = 1 : size(x,2)
                yHat(i) = yHat(i) + a(j) * x(i,j);
            end
        end
        %}
        yHat = x * a; % �s�ks pieraksts augst�k aizkoment�taj�m rindi��m
    else
        error('linreg_predict: Nav realiz�ts.');
    end
end
