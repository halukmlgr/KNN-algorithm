%% KNN algoritmasý 
%dataseti çýkýþ sýnýfý ve giriþ deðerleri olarak 2 deðiþkene atandý.
%giriþ deðerleri X olarak belirtildi.
%çýktý sýnýfýmýzda Y olarak belirtildi.
%trainKnn fonksiyonuna dataseti gönderip test verilerinin çýktý deðerleri dönüyor. 
function [ varargout ] = KNN( varargin )
        [X, Y] = divideTable(varargin{1}); 
        varargout{1} = trainKNN(X, Y, varargin{2});
end
%Giriþ matrisi ve çýkýþ vektörü arasýnda ayrým yapan fonksiyon
function [ X, Y ] = divideTable( dataset )
    if istable(dataset)
        X = table2array(dataset(:,1:end-1)); 
        %1.sutundan en sonuncu sutun haricinde alýyor
        Y = categorical(dataset.Class);
        %class alýyor kategorik hale getiriyor
    end
end
%çapraz doðrulama modeli oluþturulmasý için foksiyon tanýmlanýyor.
function [ trainingOutput ] = trainKNN( X, Y, Options )    
    % 'fold' için çapraz doðrulama modeli oluþturun.
    if ~isfield(Options, 'fold')%çapraz doðrulama yapýldýðýnda kaç parçaya ayrýlacaðý kýsým
        % Modeller parametrelere göre oluþturulmuþtur. 
        [Mdl, predictions] = NearestNeighbor( X, Y, Options );
        %model gelen x ve y' yi Near ile baþlayan foksiyona gönderiyor 
    else    
        [Mdl, ~] = NearestNeighbor(X, Y, Options);             
        [predictions] = kFoldCrossVal(Mdl, X, Options);
    end       
    % Model oluþturun istatisklikleri hesaplayýn
    trainingOutput = modelStatistics(Y, predictions);    
    % Temel eðitim ait model.
    trainingOutput.Model = Mdl;    
    % Eðitim setinin sýnýflarý tanýmlanýyor.
    trainingOutput.Categories = categories(Y);    
end
%KNN sýnýflandýrma modeli X ve Y sýnýflarýný kullanarak tahminleri döndüren
%foksiyon
function [ Mdl, Predictions ] = NearestNeighbor( X, Y, Options )    
    Mdl = fitcknn(X, Y, 'NumNeighbors', Options.k, 'Distance', 'euclidean');    
    %Oluþturulan modelin KNN algoritmasýna sokuluyor. 
    Predictions = predict(Mdl, X);
    %Tanýmlanan  modelin çýktýsýný X'e göre tahmin ediyor 
end

% Modelin 'fold' sayýsý tüm eðitim testi için çaðraz doðrulama sokuluyor.
function [ prediction ] = kFoldCrossVal( Mdl, X, Options )    
    [row, ~] = size(X);%Satýr sayýsýný tutuyor     
    if Options.fold < 1 %1 den küçükse hata veriyor 
        error('fold 1 den küçük olamaz.');
    end   
    if Options.fold < row % çapraz doðrulama için  satýr sayýsýnýndan küçük olmasý gerekiyor.
        CVMdl = crossval(Mdl, 'kfold', Options.fold);%çapraz doðrulama modeli oluþturuluyor.
    elseif Options.fold == row
        %Eðer satýr sayýsý fold sayýsýna eþit ise farklý bir çapraz doðrulama 
        %Leave-one-cut cv. uygulanýyor.
        CVMdl = crossval(Mdl, 'Leaveout', 'on');
        %farklý bir çapraz doðrulama yöntemi eðer satýr sayýsý eþit ise 
    else
       error('fold sayýsý veri kümesinde ki satýr sayýsýndan büyük olamaz.');
    end
    % Çapraz doðrulanmýþ çekirdek regresyon modeli 'CVMdl' ile 
    % çapraz doðrulanmýþ tahmini yanýtlarý döndür.
    [prediction, ~] = kfoldPredict(CVMdl);
    % Tahmin edilen classda ki deðeri veriyor.
end

% Bu fonksyionda istatistikleri hesaplýyoruz. 
function [ output ] = modelStatistics( actual, ePrediction )

    Class = categories(actual);
    %gerçek deðerleri class deðiþkenine atýyor 
    NumberOfInstanceAsClass = countcats(actual);
    %gerçek deðerlerin sayýsýný tutuyor 
    NumberOfClass = length(Class);
    %uzunluðunu tutuyor 
    ROC = zeros(NumberOfClass, 1);
    %zeros ile sýfýrlardan oluþan bir matris tanýmladýk.
    NumberOfInstance = length(actual)
    ;%gerçek deðerlerin uzunluðuna bakýlýyor
    output.NOI = NumberOfInstance;
     %karmaþýklýk matrisi oluþturuluyor 
    ConfusionMatrix = confusionmat(actual, ePrediction);  
    %true positive deðerleri 
    TP = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,2); 
    %False pozitif deðerleri.
    FP = (sum(ConfusionMatrix,1)' - diag(ConfusionMatrix)) ./ (length(actual) - sum(ConfusionMatrix,2));
    % Compute Precision.
    Precision = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,1)'; 
    %Recall.
    Recall = TP;    
    % F-Measure.
    F_Measure = (2*Precision.*Recall)./(Precision+Recall);    
    %Kappa
    pA = trace(ConfusionMatrix) / NumberOfInstance;
    pE = sum((sum(ConfusionMatrix,2).*sum(ConfusionMatrix,1)') ./ (NumberOfInstance*NumberOfInstance));
    output.Kappa = (pA-pE)/(1-pE);
    
    predict = (actual == ePrediction);
    meanPredict = mean(predict);
    sumPredict = sum(predict);
    
    % MAE
    output.MAE = 1-meanPredict;
    
    %RMSE
    output.RMSE = sqrt(1-meanPredict);
    
    % RAE
    output.RAE = sumPredict / sum(abs(meanPredict-double(predict)));
    
    % RRSE
    output.RRSE = sqrt(sumPredict / sum((meanPredict-double(predict)).^2));
    
    %TPR
    last = 0;
    TPR = zeros(length(actual), NumberOfClass);%
    for i = 1 : NumberOfClass
        first = last + 1;
        last = last + NumberOfInstanceAsClass(i);
        row = 1;
        for j = first : last
            TPR(row, i) = sum(ePrediction(first:j) == Class(i)) / NumberOfInstanceAsClass(i);
            row = row + 1;
        end
        TPR(row:end, i) = TPR(row-1, i);
    end
    
    % Roc
    for i = 1 : NumberOfClass
        ROC(i) = TP(i)*(1-FP(i));
    end
    %  accuracy tablosunun sýnýfa göre oluþturulmasý
    Class{NumberOfClass+1} = '';
    RowNames = cell(NumberOfClass+1,1);
    for i = 1 : NumberOfClass
        RowNames(i) = cellstr(int2str(i));
    end
    RowNames(end) = cellstr('Weighted Avg.');
    
    %DetailedAccuracyByClass bu kýsýmda model performansýn deðerleri
    %atanýyor 
    
    output.DetailedAccuracyByClass = table([TP; mean(TP)], [FP; mean(FP)], [Precision; mean(Precision)], ...
                                    [Recall; mean(Recall)], [F_Measure; mean(F_Measure)], [ROC; mean(ROC)], ...
                                    cellstr(Class), ...
                                    'VariableNames', {'TP', 'FP', 'Precision', 'Recall', 'FMeasure', 'ROC', ... 
                                    'Class'}, 'RowNames', RowNames);
    
    % Öngörülen çýktý.
    output.Prediction = ePrediction;
    
    %Doðru Sýnýflandýrýlmýþ Örnekler
    output.CCI = trace(ConfusionMatrix);
    
    % Yanlýþ Sýnýflandýrýlmýþ Örnekler.
    output.ICI = length(actual) - trace(ConfusionMatrix);
    
    % Doðruluk oraný
    output.AccuracyRate = output.CCI * 100 / output.NOI;
    
    % Karmaþýklýk matrisi
    CM = table;
    for i = 1 : NumberOfClass
        CM(:,i) = num2cell(ConfusionMatrix(:,i));
    end
    CM.Properties.VariableNames = Class(1:end-1);
    CM.Properties.RowNames = Class(1:end-1);
    output.ConfusionMatrix = CM;
end

