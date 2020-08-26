x = mpgX2(:,1:10);
y = mpgY;
names = mpgNames2;
plotMSE = [];
% normalizejam visus faktorus
for j = 1 : size(x,2)
    minX = min(x(:,j));
    range = max(x(:,j)) - minX;
    x(:,j) = (x(:,j) - minX) / range;
end
[n, m] = size(x);

%2. uzdevums
%k sobrid ir fiksets. Jarealize, lai sis process tiktu veikts ar
%k vertibam no 1 lidz 10.
%Japarada diagramma - kaa MSE ir atkarigs no k.
%Jasecina, cik lielu k vislabak izveleties un kurus datu kopas
%faktorus taja gadijuma vislabak lietot.

%k = 3; % tuvako kaiminu skaits
for k = 1 : 10;
allMSE = []; % saja vektora saglabasim labakos MSE no katras iteracijas, lai varetu beigâs uzzimet grafiku

currBits = false(1, m); % tekosas iteracijas labaka atrasta bitu kombinacija (logiskais indekss)
currMSE = Inf; % currBits kombinacijas novertejums (t.i., tai atbilstosa regresijas modela novertejums)

finalBits = currBits; % finalBits bus kopuma labaka atrasta bitu kombinacija. ar currBits kombinaciju sakam parmeklesanu
finalMSE = currMSE; % finalBits kombinacijas novertejums (t.i., tai atbilstosa regresijas modela novertejums)

beigas = false;
for iter = 1 : m % algoritma iteracijas numurs
    bestNewBits = NaN; %tekosanja iteracija
    bestNewMSE = Inf; %tekosaja iteracija
    for i = 1 : m 
        tryBits = currBits;
        if tryBits(i)
            continue; %pariet uz nakamo iteraciju
        end
        tryBits(i) = true;
        xUsed = x(:,tryBits);
        [~, ~, ~, MSE] = knn_loocv(xUsed, y, k);
        if MSE < bestNewMSE
            bestNewBits = tryBits;
            bestNewMSE = MSE;
        end
    end
    if bestNewMSE > currMSE
        beigas = true;
    end
    currBits = bestNewBits;
    currMSE = bestNewMSE;
    allMSE = [allMSE bestNewMSE];
    fprintf('Iterâcija #%d: MSE = %.2f\tBiti =%s\n', iter, currMSE, sprintf(' %d', currBits));
    if currMSE < finalMSE
        finalBits = currBits;
        finalMSE = currMSE;
    end
    if beigas
        break;
    end
end

fprintf('Labâkais rezultâts: MSE = %.2f ar %d faktoriem: %s\n', finalMSE, sum(finalBits), sprintf('"%s", ', names{finalBits}));
plotMSE =[plotMSE finalMSE];
end
%figure;
%plot(1:length(allMSE), allMSE);
%title('MSE atkarîba no faktoru skaita');
%xlabel('Faktoru skaits');
%ylabel('MSE');

figure;
plot(1:length(plotMSE), plotMSE);
title('MSE atkarigs no k');
xlabel('Faktoru skaits');

