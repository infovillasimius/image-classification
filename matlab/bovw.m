clear variables
clc
samples=100;
oldMethod=0;

%% Fase 0 - Preparazione cartelle
datafolder=fullfile('../data/');
trainFolder=fullfile('../data/train');
testFolder=fullfile('../data/test');

testSet = imageDatastore(testFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(testSet);
minSetCount = min(tbl{:,2}); 
samples = min(samples,minSetCount);
testSet = splitEachLabel(testSet, samples, 'randomize');

%% Fase 1 - Generazione della BAG of visual words 
if  exist('bag.mat', 'file')==0 
    dataSet = imageDatastore(trainFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
    tbl = countEachLabel(dataSet);
    minSetCount = min(tbl{:,2}); 
    samples = min(samples,minSetCount);
    dataSet = splitEachLabel(dataSet, samples, 'randomize');
    extractor = @mySIFTBagOfFeaturesExtractor;
    bag = bagOfFeatures(dataSet, 'CustomExtractor', extractor,'VocabularySize',1000,'StrongestFeatures',0.80)
    save('bag','bag');
else
    load('bag.mat');
end

%% Fase 2 - Preparazione trainingSet e validationSet
if exist('trainingSet.mat', 'file')==0 || exist('validationSet.mat', 'file')==0
    training = imageDatastore(trainFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
    tbl = countEachLabel(training);
    minSetCount = min(tbl{:,2}); 
    minSetCount = min(minSetCount,samples);
    training = splitEachLabel(training, minSetCount, 'randomize');
    [trainingSet, validationSet] = splitEachLabel(training, 0.6, 'randomize');

    save('trainingSet','trainingSet')
    save('validationSet','validationSet') 
else
    load('trainingSet')
    load('validationSet')
end

%% Fase3 - Addestramento SVM e test (Metodo standard: ImageCategoryClassifier)
if oldMethod==1
    if exist('categoryClassifier.mat', 'file')==0
        tbl = countEachLabel(training);
        minSetCount = min(tbl{:,2});
        figure
        montage(training.Files(1:minSetCount:end))

        opts = templateSVM('KernelFunction','polynomial','PolynomialOrder',3,'Solver','ISDA');

        categoryClassifier = trainImageCategoryClassifier(trainingSet, bag,'LearnerOptions',opts);
        save('categoryClassifier','categoryClassifier')
    else
        load('categoryClassifier')
    end

    if exist('categoryClassifier', 'var')==1
        trainConfMatrix = evaluate(categoryClassifier, trainingSet);
        validationConfMatrix = evaluate(categoryClassifier, validationSet);
    end
end

%% Fase 4 - Estrazione delle features
if exist('trainingFeatures.mat', 'file')==0
    trainingFeatures= encode(bag,trainingSet);
    save('trainingFeatures','trainingFeatures')
else
    load('trainingFeatures');
end

if exist('validationFeatures.mat', 'file')==0
    validationFeatures= encode(bag,validationSet);
    save('validationFeatures','validationFeatures')
else
    load('validationFeatures');
end

if exist('testFeatures.mat', 'file')==0
    testFeatures= encode(bag,testSet);
    save('testFeatures','testFeatures')
else
    load('testFeatures');
end


%% Fase 5 - Addestramento SVM multiclasse
if exist('classifier.mat', 'file')==0
    
    trainingLabels=trainingSet.Labels;
    
%    opts = templateSVM('KernelFunction','linear','Solver','SMO', 'BoxConstraint',3.2, ...
%        'IterationLimit',1e5,'KKTTolerance',0.02, 'GapTolerance',0.01);

%    opts = templateSVM('KernelFunction','rbf','Solver','SMO','BoxConstraint',4, ...
%        'IterationLimit',2e5, 'KKTTolerance',0.02, 'GapTolerance',0.01);
    
    opts = templateSVM('KernelFunction','polynomial','PolynomialOrder',3,'Solver','ISDA');

%
%    opts = templateSVM('KernelFunction','polynomial','PolynomialOrder',3, ...
%     'Solver', 'ISDA','BoxConstraint',1,'ClipAlphas',true, 'DeltaGradientTolerance',0.01, ...
%     'GapTolerance',0.0, 'KernelOffset',0.5, 'KernelScale',1.05, 'KKTTolerance',0.1, ...
%     'OutlierFraction',0.05, 'Standardize',false, 'Verbose',0);


    classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', opts, 'Coding', 'onevsall');
    save('classifier','classifier')
else
    load('classifier');
end

%% Risultati Training SET
predictedTrainingLabels = predict(classifier, trainingFeatures );
trainingLabels = trainingSet.Labels;
% Tabulate the results using a confusion matrix.
confTrainingMat = confusionmat(trainingLabels, predictedTrainingLabels);

% Convert confusion matrix into percentage form
confTrainingMat = bsxfun(@rdivide,confTrainingMat,sum(confTrainingMat,2));

% Display the mean accuracy
trainingSetResult = mean(diag(confTrainingMat))

%% Risultati Validation SET
predictedLabels = predict(classifier, validationFeatures);
validationLabels = validationSet.Labels;
% Tabulate the results using a confusion matrix.
confMat = confusionmat(validationLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Display the mean accuracy
validationSetResult = mean(diag(confMat))

%% Risultati Test SET
predictedTestLabels = predict(classifier, testFeatures);
testLabels = testSet.Labels;
% Tabulate the results using a confusion matrix.
confTestMat = confusionmat(testLabels, predictedTestLabels);

% Convert confusion matrix into percentage form
confTestMat = bsxfun(@rdivide,confTestMat,sum(confTestMat,2));
testSetResult = mean(diag(confTestMat))
