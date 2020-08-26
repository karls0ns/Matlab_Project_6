function a = linreg(z, y, degree)
% funkcija nosaka parametrus a attiec�g�s pak�pes polinoma regresijas modelim
    m = size(z,2);
    % noslinkosim - realiz�sim visu tikai viena faktora jebkuras pak�pes
    % polinoma regresijai un daudzfaktoru pirm�s pak�pes polinoma regresijai,
    % bet daudzfaktoru gad�jumam ar pak�pi >1 izvad�sim k��das pazi�ojumu
    if (degree <= 1) || (m <= 1)
        % pirmaj� kolonn� visi vieninieki
        x = ones(size(z,1), 1);
        % p�r�j�s kolonnas ir z v�rt�bas celtas pieaugo�� pak�p� (no 1 l�dz degree)
        for i = 1 : degree
            x = [x z .^ i];
        end
        a = x \ y; % nosak�m parametrus a
    else
        error('linreg: Nav realiz�ts.');
    end
end
