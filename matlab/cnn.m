clear variables
clc

imageFolder = fullfile('data/train');
testImageFolder = fullfile('data/test');

trainSet = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
testSet = imageDatastore(testImageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

tbl = countEachLabel(trainSet);
tbl2 = countEachLabel(testSet);
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 
minSetCount2 = min(tbl2{:,2});

maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);
minSetCount2 = min(maxNumImages,minSetCount2);

% Use splitEachLabel method to trim the set.
trainSet = splitEachLabel(trainSet, minSetCount, 'randomize');
%testSet = splitEachLabel(testSet, minSetCount2, 'randomize');

% Display an image for each category
%figure
%montage(trainSet.Files(1:minSetCount:end))

% Load pretrained network
net = resnet50();

[trainingSet, validationSet] = splitEachLabel(trainSet, 0.9, 'randomize');

% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, validationSet, 'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.

% opts = templateSVM('KernelFunction','linear','Solver','ISDA');

%    opts = templateSVM('KernelFunction','rbf','Solver','SMO');

opts = templateSVM('KernelFunction','polynomial','PolynomialOrder',3,'Solver','ISDA');

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', opts, 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Extract validation and test features using the CNN
validationFeatures = activations(net, augmentedValidationSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedValidationLabels = predict(classifier, validationFeatures, 'ObservationsIn', 'columns');
predictedTestLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
validationLabels = validationSet.Labels;
testLabels = testSet.Labels;

% Validation Set Results
validationConfMat = confusionmat(validationLabels, predictedValidationLabels);
validationConfMat = bsxfun(@rdivide,validationConfMat,sum(validationConfMat,2))
mean(diag(validationConfMat))

% Test Set Results
testConfMat = confusionmat(testLabels, predictedTestLabels);
testConfMat = bsxfun(@rdivide,testConfMat,sum(testConfMat,2))
mean(diag(testConfMat))

