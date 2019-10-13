% function [desc] = detect_features(img);
%
% Detect and describe features for an image 
%
% IN: an image 
% OUT: SIFT object  containing detected LoG features described with SIFT descriptors.
%      
% The output contains structure "desc" with fileds:
%
%  desc.r    ... row index of each feature
%  desc.c    ... column index of each feature
%  desc.rad  ... radius (scale) of each feauture
%  desc.sift ... 128d SIFT descriptor for each feature
%
% Josef.Sivic@ens.fr
% 7/11/2009
%
% Modified by Lamberto Ballan - 2/9/2013
% Modified by Antonello Meloni - 14/08/2019

function [desc] = detectSIFTFeatures(Im,detectOnly,grid)   
    
    % detector paramteres
    sigma       = 2;              % initial scale
    k           = sqrt(sqrt(2));  % scale step
    sigma_final = 16;             % final scale
    threshold   = 0.005;          % squared response threshold

    % descriptor parameters
    enlarge_factor = 2; % enlarge the size of the features to make them more distinctive

    Im = mean(double(Im),3)./max(double(Im(:)));

    if exist ('grid', 'var')
        % use points provided
        r=grid(:,2);
        c=grid(:,1);
        rad=grid(:,3);
    else
        % compute features (LoG)
        [r, c, rad] = BlobDetector(Im, sigma, k, sigma_final, threshold);
    end
    
    if (exist('detectOnly','var')==0 || detectOnly==false)
        % describe features
        circles = [c r rad];  

        sift_arr = find_sift(Im, circles, enlarge_factor);

        % convert to single to save space
        desc = struct('sift',sift_arr,'r',r,'c',c,'rad',rad);
    else
        desc = struct('sift',[],'r',r,'c',c,'rad',rad);
    end
end


%% Scale invariant blob detector solution code
%% Adapted by Svetlana Lazebnik and Josef Sivic based on code by Serhat Tekin 
%% UNC Chapel Hill, COMP 776 (Spring 2008)
%% ENS Paris, Object recognition and Computer Vision (Fall 2009)
%%
%% Usage:  [r, c, rad] = BlobDetector(im, sigma, k, sigma_final, threshold)
%%
%% Arguments:
%%            im     - input image
%%            sigma  - initial scale
%%            k   - scale multiplication constant
%%            sigma_final - largest scale to process
%%            threshold - Laplacian threshold
%%
%% Returns:
%%            r      - row coordinates of blob centers (y)
%%            c      - column coordinates of blob centers (x)
%%            rad    - circular blob radius
%%

function [r, c, rad] = BlobDetector(im, sigma, k, sigma_final, threshold)

    if size(im,3)>1
        im = mean(im,3)/255;
    end

    n = ceil((log(sigma_final) - log(sigma))/log(k)); % number of scale iterations

    % allocate state space
    [h, w] = size(im); % h, w => height and width of the state space
    scaleSpace = zeros(h, w, n);

    % generate the Laplacian of Gaussian for the first scale level
    filt_size = 2*ceil(3*sigma)+1;  % important: to avoid "shifting" artifacts, make sure the kernel size is odd!
    LoG =  sigma^2 * fspecial('log', filt_size, sigma);

    % generate the responses for the remaining levels
    imRes = im;
    for i = 1:n
        imFiltered = imfilter(imRes, LoG, 'same', 'replicate'); % filter the image with LoG
        % note that no scale normalization is needed: the fact that the filter
        % remains the same size while the image is downsampled ensures that the
        % response of the filter is scale-invariant
        imFiltered = imFiltered .^ 2; % save square of the response for current level

        % upsample the LoG response to the original image size
        scaleSpace(:,:,i) = imresize(imFiltered, size(im), 'bicubic'); % bilinear supersampling will result in a loss of spatial resolution
        if i < n        
            imRes = imresize(im, 1/(k^i), 'bicubic');
        end
    end

    % perform non-maximum suppression for each scale-space slice
    supprSize = 3;
    maxSpace = zeros(h, w, n);
    for i = 1:n
        maxSpace(:,:,i) = ordfilt2(scaleSpace(:,:,i), supprSize^2, ones(supprSize));
    end

    % non-maximum suppression between scales and threshold
    for i = 1:n
        maxSpace(:,:,i) = max(maxSpace(:,:,max(i-1,1):min(i+1,n)),[],3);
    end
    maxSpace = maxSpace .* (maxSpace == scaleSpace);

    r = [];   
    c = [];   
    rad = [];
    for i=1:n
        [rows, cols] = find(maxSpace(:,:,i) >= threshold);
        numBlobs = length(rows);
        radii =  sigma * k^(i-1) * sqrt(2); 
        radii = repmat(radii, numBlobs, 1);
        r = [r; rows];
        c = [c; cols];
        rad = [rad; radii];
    end

end


function sift_arr = find_sift(I, circles, enlarge_factor)
%%
%% Compute non-rotation-invariant SIFT descriptors of a set of circles 
%% I is the image
%% circles is an Nx3 array where N is the number of circles, where the
%%    first column is the x-coordinate (column), the second column is the y-coordinate (row),
%%    and the third column is the radius
%% enlarge_factor is by how much to enarge the radius of the circle before
%%    computing the descriptor (a factor of 1.5 or larger is usually necessary
%%    for best performance)
%% The output is an Nx128 array of SIFT descriptors
%% (c) Lana Lazebnik
%%

    if ndims(I) == 3
        I = im2double(rgb2gray(I));
    else
        I = im2double(I);
    end

    % parameters (default SIFT size)
    num_angles = 8;
    num_bins = 4;
    num_samples = num_bins * num_bins;
    alpha = 9; % smoothing for orientation histogram

    if nargin < 3
        enlarge_factor = 1.5;
    end

    angle_step = 2 * pi / num_angles;
    angles = 0:angle_step:2*pi;
    angles(num_angles+1) = []; % bin centers

    [hgt, wid] = size(I);
    num_pts = size(circles,1);

    sift_arr = zeros(num_pts, num_samples * num_angles);

    % edge image
    sigma_edge = 1;

    [G_X,G_Y]=gen_dgauss(sigma_edge);
    I_X = filter2(G_X, I, 'same'); % vertical edges
    I_Y = filter2(G_Y, I, 'same'); % horizontal edges
    I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
    I_theta = atan2(I_Y,I_X);
    I_theta(isnan(I_theta)) = 0; % necessary????

    % make default grid of samples (centered at zero, width 2)
    interval = 2/num_bins:2/num_bins:2;
    interval = interval - (1/num_bins + 1);
    [grid_x, grid_y] = meshgrid(interval, interval);
    grid_x = reshape(grid_x, [1 num_samples]);
    grid_y = reshape(grid_y, [1 num_samples]);

    % make orientation images
    I_orientation = zeros(hgt, wid, num_angles);
    % for each histogram angle
    for a=1:num_angles    
        % compute each orientation channel
        tmp = cos(I_theta - angles(a)).^alpha;
        tmp = tmp .* (tmp > 0);

        % weight by magnitude
        I_orientation(:,:,a) = tmp .* I_mag;
    end

    % for all circles
    for i=1:num_pts
        cx = circles(i,1);
        cy = circles(i,2);
        r = circles(i,3) * enlarge_factor;

        % find coordinates of sample points (bin centers)
        grid_x_t = grid_x * r + cx;
        grid_y_t = grid_y * r + cy;
        grid_res = grid_y_t(2) - grid_y_t(1);

        % find window of pixels that contributes to this descriptor
        x_lo = floor(max(cx - r - grid_res/2, 1));
        x_hi = ceil(min(cx + r + grid_res/2, wid));
        y_lo = floor(max(cy - r - grid_res/2, 1));
        y_hi = ceil(min(cy + r + grid_res/2, hgt));

        % find coordinates of pixels
        [grid_px, grid_py] = meshgrid(x_lo:x_hi,y_lo:y_hi);
        num_pix = numel(grid_px);
        grid_px = reshape(grid_px, [num_pix 1]);
        grid_py = reshape(grid_py, [num_pix 1]);

        % find (horiz, vert) distance between each pixel and each grid sample
        dist_px = abs(repmat(grid_px, [1 num_samples]) - repmat(grid_x_t, [num_pix 1])); 
        dist_py = abs(repmat(grid_py, [1 num_samples]) - repmat(grid_y_t, [num_pix 1])); 

        % find weight of contribution of each pixel to each bin
        weights_x = dist_px/grid_res;
        weights_x = (1 - weights_x) .* (weights_x <= 1);
        weights_y = dist_py/grid_res;
        weights_y = (1 - weights_y) .* (weights_y <= 1);
        weights = weights_x .* weights_y;

        % make sift descriptor
        curr_sift = zeros(num_angles, num_samples);
        for a = 1:num_angles
            tmp = reshape(I_orientation(y_lo:y_hi,x_lo:x_hi,a),[num_pix 1]);        
            tmp = repmat(tmp, [1 num_samples]);
            curr_sift(a,:) = sum(tmp .* weights);
        end    
        sift_arr(i,:) = reshape(curr_sift, [1 num_samples * num_angles]);    

    end


    %%
    %% normalize the SIFT descriptors more or less as described in Lowe (2004)
    %%
    tmp = sqrt(sum(sift_arr.^2, 2));
    normalize_ind = find(tmp > 1);

    sift_arr_norm = sift_arr(normalize_ind,:);
    sift_arr_norm = sift_arr_norm ./ repmat(tmp(normalize_ind,:), [1 size(sift_arr,2)]);

    % suppress large gradients
    sift_arr_norm(find(sift_arr_norm > 0.2)) = 0.2;

    % finally, renormalize to unit length
    tmp = sqrt(sum(sift_arr_norm.^2, 2));
    sift_arr_norm = sift_arr_norm ./ repmat(tmp, [1 size(sift_arr,2)]);

    sift_arr(normalize_ind,:) = sift_arr_norm;

end


function [GX,GY]=gen_dgauss(sigma)

    f_wid = 4 * floor(sigma);
    G = normpdf(-f_wid:f_wid,0,sigma);
    G = G' * G;
    [GX,GY] = gradient(G); 

    GX = GX * 2 ./ sum(sum(abs(GX)));
    GY = GY * 2 ./ sum(sum(abs(GY)));

end