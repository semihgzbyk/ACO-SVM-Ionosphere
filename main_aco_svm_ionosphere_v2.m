%% =========================================================================
%% ACO ile SVM Hiperparametre Optimizasyonu - Ionosphere (v2)
%% Düzeltmeler:
%%   1. Birleşik 3B Feromon Matrisi (parametreler artık birlikte seçiliyor)
%%   2. MMAS — Min/Max feromon sınırlandırması eklendi
%%   3. Sezgisellik (beta) matrisi eklendi
%%   4. Erken durdurma (patience) kriteri eklendi
%%   5. feromonSec -> feromonSeçimi fonksiyonu, alfa+beta birlikte kullanılıyor
%% =========================================================================

clc; clear; close all;
rng(42);

%% ─────────────────────────────────────────────────────────────────────────
%% 1. VERİ YÜKLEME
%% ─────────────────────────────────────────────────────────────────────────
load ionosphere

X = X;
y = categorical(Y);

% Sabit (sıfır varyans) sütunları çıkar
sabitSutun = std(X, 0, 1) == 0;
X(:, sabitSutun) = [];

fprintf('Örnek sayısı  : %d\n', size(X,1));
fprintf('Özellik sayısı: %d\n\n', size(X,2));

%% ─────────────────────────────────────────────────────────────────────────
%% 2. TRAIN / TEST AYRIMI
%% ─────────────────────────────────────────────────────────────────────────
cv = cvpartition(y, 'HoldOut', 0.20);

X_train = X(training(cv), :);  y_train = y(training(cv));
X_test  = X(test(cv),     :);  y_test  = y(test(cv));

fprintf('Train: %d | Test: %d\n\n', size(X_train,1), size(X_test,1));

%% ─────────────────────────────────────────────────────────────────────────
%% 3. CROSS-VALIDATION PARTİSYONU
%% ─────────────────────────────────────────────────────────────────────────
kFold   = 5;
cvTrain = cvpartition(y_train, 'KFold', kFold);

%% ─────────────────────────────────────────────────────────────────────────
%% 4. HİPERPARAMETRE ARAMA UZAYI
%% ─────────────────────────────────────────────────────────────────────────
boxDegerleri        = logspace(-2, 2.5, 25);   % BoxConstraint
kernelDegerleri     = logspace(-1, 1.5, 25);   % KernelScale
standardizeDegerleri = [true false];           % Standardize

nBox  = numel(boxDegerleri);
nKer  = numel(kernelDegerleri);
nStd  = numel(standardizeDegerleri);

%% ─────────────────────────────────────────────────────────────────────────
%% 5. ACO PARAMETRELERİ
%% ─────────────────────────────────────────────────────────────────────────
karincaSayisi  = 20;
iterasyonSayisi = 30;

alfa       = 1;     % Feromon ağırlığı
beta       = 2;     % Sezgisellik ağırlığı
buharlasma = 0.15;  % Buharlaşma oranı
Q          = 1;     % Ödül sabiti

% ── MMAS Sınırları ──────────────────────────────────────────────────────
tau_min = 0.1;   % Yolların tamamen unutulmasını engeller
tau_max = 10.0;  % Bir yolun aşırı baskın olmasını engeller

% ── Erken Durdurma ──────────────────────────────────────────────────────
patience        = 8;   % Kaç iterasyon iyileşme olmazsa dur
enIyiIyilesme  = 0;   % Sayaç

%% ─────────────────────────────────────────────────────────────────────────
%% 6. BİRLEŞİK 3B FEROMON & SEZGİSELLİK MATRİSİ
%%
%%  DÜZELTME: Önceki kodda Box, Kernel, Standardize için 3 ayrı
%%  bağımsız vektör vardı. Bu, parametreler arasındaki etkileşimi
%%  (C ile KernelScale kuvvetli korelasyon) tamamen görmezden geliyordu.
%%  Şimdi tek bir [nBox × nKer × nStd] matrisi kullanıyoruz; her hücre
%%  o parametre üçlüsünü birlikte temsil ediyor.
%% ─────────────────────────────────────────────────────────────────────────
feromon     = ones(nBox, nKer, nStd);           % Başlangıç: eşit koku
sezgisellik = ones(nBox, nKer, nStd);           % Başlangıç: eşit sezgi

%% ─────────────────────────────────────────────────────────────────────────
%% 7. BAŞLANGIÇ NOKTASI (Classification Learner çıktısı)
%% ─────────────────────────────────────────────────────────────────────────
enIyiBox         = 1;
enIyiKernel      = 5.7;
enIyiStandardize = true;

enIyiHata = svmHata(X_train, y_train, cvTrain, ...
                    enIyiBox, enIyiKernel, enIyiStandardize);

fprintf('Başlangıç modeli (Classification Learner):\n');
fprintf('  BoxConstraint = %.4f | KernelScale = %.4f | Standardize = %d\n', ...
        enIyiBox, enIyiKernel, enIyiStandardize);
fprintf('  CV Başarısı   = %.2f%%\n\n', (1-enIyiHata)*100);

%% ─────────────────────────────────────────────────────────────────────────
%% 8. ACO ANA DÖNGÜSÜ
%% ─────────────────────────────────────────────────────────────────────────
gecmisBasari = zeros(iterasyonSayisi, 1);
gercekIter   = iterasyonSayisi;   % Erken durmaya göre güncellenir

for iter = 1:iterasyonSayisi

    fprintf('\n==============================\n');
    fprintf('İTERASYON %d / %d\n', iter, iterasyonSayisi);
    fprintf('==============================\n');

    karincaIdx  = zeros(karincaSayisi, 3);   % [boxIdx kernelIdx stdIdx]
    karincaHata = zeros(karincaSayisi, 1);

    iterIyilesti = false;

    for k = 1:karincaSayisi

        %% Birleşik olasılık hesabı (alfa + beta birlikte)
        %%   DÜZELTME: Eski kodda sadece feromon^alfa kullanılıyordu.
        %%   Şimdi (feromon^alfa) * (sezgisellik^beta) — standart ACO formülü.
        olasilik3B = (feromon .^ alfa) .* (sezgisellik .^ beta);
        olasilik1B = olasilik3B(:) ./ sum(olasilik3B(:));

        % Rulet tekerleği seçimi
        secim = randsample(1:numel(olasilik1B), 1, true, olasilik1B);
        [bIdx, kIdx, sIdx] = ind2sub([nBox nKer nStd], secim);

        box        = boxDegerleri(bIdx);
        kernel     = kernelDegerleri(kIdx);
        standardize = standardizeDegerleri(sIdx);

        karincaIdx(k, :) = [bIdx kIdx sIdx];

        % Model değerlendirme
        hata   = svmHata(X_train, y_train, cvTrain, box, kernel, standardize);
        basari = (1 - hata) * 100;

        karincaHata(k) = hata;

        % Sezgiselliği güncelle (o konumun kalitesini öğren)
        sezgisellik(bIdx, kIdx, sIdx) = max(1 - hata, 0.0001);

        fprintf('Karınca %2d | Box=%8.4f | Kernel=%8.4f | Std=%d | Başarı=%.2f%%', ...
                k, box, kernel, standardize, basari);

        if hata < enIyiHata
            enIyiHata        = hata;
            enIyiBox         = box;
            enIyiKernel      = kernel;
            enIyiStandardize = standardize;
            iterIyilesti     = true;
            fprintf('  <-- Yeni en iyi!');
        end
        fprintf('\n');
    end

    %% ── Feromon Güncelleme (MMAS) ──────────────────────────────────────
    %  1. Buharlaşma
    feromon = (1 - buharlasma) .* feromon;

    %  2. Koku bırakma (başarı orantılı)
    for k = 1:karincaSayisi
        bIdx = karincaIdx(k,1);
        kIdx = karincaIdx(k,2);
        sIdx = karincaIdx(k,3);
        feromon(bIdx, kIdx, sIdx) = ...
            feromon(bIdx, kIdx, sIdx) + Q / (karincaHata(k) + eps);
    end

    %  3. MMAS sınırlandırması
    %%   DÜZELTME: Önceki kodda bu yoktu. Feromon sonsuza gidebilir ya
    %%   da sıfırlanabilirdi; algoritma tek bir noktaya kilitlenirdi.
    feromon = max(tau_min, min(tau_max, feromon));

    gecmisBasari(iter) = (1 - enIyiHata) * 100;

    fprintf('\n  → İterasyon sonu CV Başarısı: %.2f%%\n', gecmisBasari(iter));

    %% ── Erken Durdurma ─────────────────────────────────────────────────
    %%   DÜZELTME: Önceki kodda yoktu. Gereksiz hesaplamayı önler.
    if iterIyilesti
        enIyiIyilesme = 0;
    else
        enIyiIyilesme = enIyiIyilesme + 1;
        fprintf('  [Patience: %d / %d]\n', enIyiIyilesme, patience);
        if enIyiIyilesme >= patience
            fprintf('\n  *** Erken durdurma: %d iterasyondur iyileşme yok. ***\n', patience);
            gercekIter = iter;
            break;
        end
    end
end

%% ─────────────────────────────────────────────────────────────────────────
%% 9. NİHAİ MODEL EĞİTİMİ & TEST
%% ─────────────────────────────────────────────────────────────────────────
finalModel = fitcsvm( ...
    X_train, y_train, ...
    'KernelFunction', 'gaussian', ...
    'BoxConstraint',  enIyiBox, ...
    'KernelScale',    enIyiKernel, ...
    'Standardize',    enIyiStandardize);

y_pred    = predict(finalModel, X_test);
testBasari = mean(y_pred == y_test) * 100;

fprintf('\n==============================\n');
fprintf('FINAL SONUÇ\n');
fprintf('==============================\n');
fprintf('En iyi BoxConstraint  = %.6f\n', enIyiBox);
fprintf('En iyi KernelScale    = %.6f\n', enIyiKernel);
fprintf('En iyi Standardize    = %d\n',   enIyiStandardize);
fprintf('Eğitim CV Başarısı    = %.2f%%\n', (1-enIyiHata)*100);
fprintf('Test Başarısı         = %.2f%%\n', testBasari);
fprintf('Gerçekleşen İterasyon = %d / %d\n', gercekIter, iterasyonSayisi);

%% ─────────────────────────────────────────────────────────────────────────
%% 10. GÖRSELLEŞTİRME
%% ─────────────────────────────────────────────────────────────────────────
aktifIter = gecmisBasari(1:gercekIter);

figure('Position', [50 50 1200 480]);

subplot(1,2,1);
plot(1:gercekIter, aktifIter, '-o', 'LineWidth', 2, 'MarkerSize', 7);
xlabel('İterasyon');
ylabel('En iyi CV Başarısı (%)');
title('ACO-SVM Yakınsama Grafiği');
ylim([max(0, min(aktifIter)-3), 100]);
grid on;

subplot(1,2,2);
confusionchart(y_test, y_pred);
title(sprintf('Test Confusion Matrix — Başarı: %.2f%%', testBasari));

sgtitle('ACO-SVM Ionosphere — v2 (Birleşik Feromon + MMAS + Sezgisellik + Erken Durdurma)', ...
        'FontSize', 11, 'FontWeight', 'bold');

%% ─────────────────────────────────────────────────────────────────────────
%% 11. SONUÇ TABLOSU
%% ─────────────────────────────────────────────────────────────────────────
sonucTablosu = table( ...
    (1:gercekIter)', ...
    gecmisBasari(1:gercekIter), ...
    'VariableNames', {'Iterasyon','EnIyiCVBasarisi'});
disp(sonucTablosu);

%% =========================================================================
%% YEREL FONKSİYONLAR
%% =========================================================================

function hata = svmHata(X, y, cv, box, kernel, standardize)
% Verilen hiperparametrelerle SVM eğitir ve 5-Fold CV hatasını döndürür.
    try
        model = fitcsvm( ...
            X, y, ...
            'KernelFunction', 'gaussian', ...
            'BoxConstraint',  box, ...
            'KernelScale',    kernel, ...
            'Standardize',    standardize);

        cvModel = crossval(model, 'CVPartition', cv);
        hata    = kfoldLoss(cvModel);
    catch
        hata = inf;
    end
end