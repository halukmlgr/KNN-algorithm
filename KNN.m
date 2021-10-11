%% KNN algoritmas� 
%dataseti ��k�� s�n�f� ve giri� de�erleri olarak 2 de�i�kene atand�.
%giri� de�erleri X olarak belirtildi.
%��kt� s�n�f�m�zda Y olarak belirtildi.
%trainKnn fonksiyonuna dataseti g�nderip test verilerinin ��kt� de�erleri d�n�yor. 
function [ varargout ] = KNN( varargin )
        [X, Y] = divideTable(varargin{1}); 
        varargout{1} = trainKNN(X, Y, varargin{2});
end
%Giri� matrisi ve ��k�� vekt�r� aras�nda ayr�m yapan fonksiyon
function [ X, Y ] = divideTable( dataset )
    if istable(dataset)
        X = table2array(dataset(:,1:end-1)); 
        %1.sutundan en sonuncu sutun haricinde al�yor
        Y = categorical(dataset.Class);
        %class al�yor kategorik hale getiriyor
    end
end
%�apraz do�rulama modeli olu�turulmas� i�in foksiyon tan�mlan�yor.
function [ trainingOutput ] = trainKNN( X, Y, Options )    
    % 'fold' i�in �apraz do�rulama modeli olu�turun.
    if ~isfield(Options, 'fold')%�apraz do�rulama yap�ld���nda ka� par�aya ayr�laca�� k�s�m
        % Modeller parametrelere g�re olu�turulmu�tur. 
        [Mdl, predictions] = NearestNeighbor( X, Y, Options );
        %model gelen x ve y' yi Near ile ba�layan foksiyona g�nderiyor 
    else    
        [Mdl, ~] = NearestNeighbor(X, Y, Options);             
        [predictions] = kFoldCrossVal(Mdl, X, Options);
    end       
    % Model olu�turun istatisklikleri hesaplay�n
    trainingOutput = modelStatistics(Y, predictions);    
    % Temel e�itim ait model.
    trainingOutput.Model = Mdl;    
    % E�itim setinin s�n�flar� tan�mlan�yor.
    trainingOutput.Categories = categories(Y);    
end
%KNN s�n�fland�rma modeli X ve Y s�n�flar�n� kullanarak tahminleri d�nd�ren
%foksiyon
function [ Mdl, Predictions ] = NearestNeighbor( X, Y, Options )    
    Mdl = fitcknn(X, Y, 'NumNeighbors', Options.k, 'Distance', 'euclidean');    
    %Olu�turulan modelin KNN algoritmas�na sokuluyor. 
    Predictions = predict(Mdl, X);
    %Tan�mlanan  modelin ��kt�s�n� X'e g�re tahmin ediyor 
end

% Modelin 'fold' say�s� t�m e�itim testi i�in �a�raz do�rulama sokuluyor.
function [ prediction ] = kFoldCrossVal( Mdl, X, Options )    
    [row, ~] = size(X);%Sat�r say�s�n� tutuyor     
    if Options.fold < 1 %1 den k���kse hata veriyor 
        error('fold 1 den k���k olamaz.');
    end   
    if Options.fold < row % �apraz do�rulama i�in  sat�r say�s�n�ndan k���k olmas� gerekiyor.
        CVMdl = crossval(Mdl, 'kfold', Options.fold);%�apraz do�rulama modeli olu�turuluyor.
    elseif Options.fold == row
        %E�er sat�r say�s� fold say�s�na e�it ise farkl� bir �apraz do�rulama 
        %Leave-one-cut cv. uygulan�yor.
        CVMdl = crossval(Mdl, 'Leaveout', 'on');
        %farkl� bir �apraz do�rulama y�ntemi e�er sat�r say�s� e�it ise 
    else
       error('fold say�s� veri k�mesinde ki sat�r say�s�ndan b�y�k olamaz.');
    end
    % �apraz do�rulanm�� �ekirdek regresyon modeli 'CVMdl' ile 
    % �apraz do�rulanm�� tahmini yan�tlar� d�nd�r.
    [prediction, ~] = kfoldPredict(CVMdl);
    % Tahmin edilen classda ki de�eri veriyor.
end

% Bu fonksyionda istatistikleri hesapl�yoruz. 
function [ output ] = modelStatistics( actual, ePrediction )

    Class = categories(actual);
    %ger�ek de�erleri class de�i�kenine at�yor 
    NumberOfInstanceAsClass = countcats(actual);
    %ger�ek de�erlerin say�s�n� tutuyor 
    NumberOfClass = length(Class);
    %uzunlu�unu tutuyor 
    ROC = zeros(NumberOfClass, 1);
    %zeros ile s�f�rlardan olu�an bir matris tan�mlad�k.
    NumberOfInstance = length(actual)
    ;%ger�ek de�erlerin uzunlu�una bak�l�yor
    output.NOI = NumberOfInstance;
     %karma��kl�k matrisi olu�turuluyor 
    ConfusionMatrix = confusionmat(actual, ePrediction);  
    %true positive de�erleri 
    TP = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,2); 
    %False pozitif de�erleri.
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
    %  accuracy tablosunun s�n�fa g�re olu�turulmas�
    Class{NumberOfClass+1} = '';
    RowNames = cell(NumberOfClass+1,1);
    for i = 1 : NumberOfClass
        RowNames(i) = cellstr(int2str(i));
    end
    RowNames(end) = cellstr('Weighted Avg.');
    
    %DetailedAccuracyByClass bu k�s�mda model performans�n de�erleri
    %atan�yor 
    
    output.DetailedAccuracyByClass = table([TP; mean(TP)], [FP; mean(FP)], [Precision; mean(Precision)], ...
                                    [Recall; mean(Recall)], [F_Measure; mean(F_Measure)], [ROC; mean(ROC)], ...
                                    cellstr(Class), ...
                                    'VariableNames', {'TP', 'FP', 'Precision', 'Recall', 'FMeasure', 'ROC', ... 
                                    'Class'}, 'RowNames', RowNames);
    
    % �ng�r�len ��kt�.
    output.Prediction = ePrediction;
    
    %Do�ru S�n�fland�r�lm�� �rnekler
    output.CCI = trace(ConfusionMatrix);
    
    % Yanl�� S�n�fland�r�lm�� �rnekler.
    output.ICI = length(actual) - trace(ConfusionMatrix);
    
    % Do�ruluk oran�
    output.AccuracyRate = output.CCI * 100 / output.NOI;
    
    % Karma��kl�k matrisi
    CM = table;
    for i = 1 : NumberOfClass
        CM(:,i) = num2cell(ConfusionMatrix(:,i));
    end
    CM.Properties.VariableNames = Class(1:end-1);
    CM.Properties.RowNames = Class(1:end-1);
    output.ConfusionMatrix = CM;
end

