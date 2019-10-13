function [features, featureMetrics, varargout] = mySIFTBagOfFeaturesExtractor(I)
% This function implements the Sift feature extraction used in
% bagOfFeatures 
 
%% Step 1: Preprocess the Image

% Convert I to grayscale.
[height,width,numChannels] = size(I);


%% Step 2: Select Point Locations for Feature Extraction
% Here, a regular spaced grid of point locations is created over I. This
% allows for dense feature extraction. 

% Define a regular grid over I.
gridStep = 8; % in pixels
gridX = 1:gridStep:width;
gridY = 1:gridStep:height;

[x,y,rd] = meshgrid(gridX, gridY, 5);
gridSIFTLocations = [x(:) y(:) rd(:);x(:) y(:) rd(:)+5];



%% Step 3: Extract features
% Extract features from the selected point locations. 
SIFTFeatures = detectSIFTFeatures(I,false);
GridSIFTFeatures = detectSIFTFeatures(I,false, gridSIFTLocations);
features = [single(GridSIFTFeatures.sift); single(SIFTFeatures.sift)];

%% Step 4: Compute the Feature Metric
featureMetrics = var(features,[],2);

% Return feature location information
if nargout > 2
    varargout{1} = [SIFTFeatures.c SIFTFeatures.r ; GridSIFTFeatures.c GridSIFTFeatures.r];
end


