function imageForgeryDetection1(originalImgPath, forgedImgPath)
    if nargin < 2
        originalImgPath = 'original1.jpg';
        forgedImgPath = 'forged1.jpg';
        fprintf('⚠️ No input arguments provided. Using default image paths.\n');
    end

    originalImg = imread(originalImgPath);
    forgedImg = imread(forgedImgPath);

    if ~isequal(size(originalImg), size(forgedImg))
        forgedImg = imresize(forgedImg, [size(originalImg,1), size(originalImg,2)]);
        warning('Images were different sizes. Resized forged image to match original.');
    end

    figure('Name', 'Original vs Forged Images', 'NumberTitle', 'off');
    subplot(1,2,1); imshow(originalImg); title('Original Image');
    subplot(1,2,2); imshow(forgedImg); title('Forged Image');

    % 1. Pixel Difference Analysis
    pixelAcc = analyzePixelDifferences(originalImg, forgedImg);

    % 2. Error Level Analysis
    if isJpeg(originalImgPath) && isJpeg(forgedImgPath)
        elaAcc = performELA(originalImg, forgedImg);
    else
        fprintf('\n⚠️ ELA skipped: Only supported for JPEG images.\n');
        elaAcc = 0;
    end

    % 3. Noise Variance Analysis
    if size(originalImg, 3) == 3, originalGray = rgb2gray(originalImg);
    else, originalGray = originalImg; end

    if size(forgedImg, 3) == 3, forgedGray = rgb2gray(forgedImg);
    else, forgedGray = forgedImg; end

    noiseAcc = analyzeNoiseVariance(originalGray, forgedGray);

    % 4. Copy-Move Detection
    copyMoveAcc = detectCopyMove(forgedImg);

    % 5. Metadata Comparison
    metadataAcc = compareMetadata(originalImgPath, forgedImgPath);

    % Final Accuracy Plot
    plotAccuracyComparison(pixelAcc, elaAcc, noiseAcc, copyMoveAcc, metadataAcc);
end

function accuracy = analyzePixelDifferences(original, forged)
    diffImg = imabsdiff(original, forged);
    threshold = 20;

    if size(diffImg, 3) == 3
        diffMask = (diffImg(:,:,1) > threshold) | ...
                   (diffImg(:,:,2) > threshold) | ...
                   (diffImg(:,:,3) > threshold);
    else
        diffMask = diffImg > threshold;
    end

    figure('Name', 'Pixel Difference Analysis', 'NumberTitle', 'off');
    subplot(1,2,1); imshow(diffImg); title('Absolute Difference');
    subplot(1,2,2); imshow(diffMask); title('Binary Difference (Thresholded)');

    changedPixels = sum(diffMask(:));
    totalPixels = numel(diffMask);
    accuracy = 100 * (1 - changedPixels / totalPixels);

    fprintf('\nPixel Difference Analysis:\n');
    fprintf('Detected changed pixels: %d\n', changedPixels);
    fprintf('Accuracy: %.2f%%\n', accuracy);
end

function result = isJpeg(filepath)
    [~,~,ext] = fileparts(filepath);
    result = strcmpi(ext, '.jpg') || strcmpi(ext, '.jpeg');
end

function accuracy = performELA(originalImg, forgedImg)
    imwrite(originalImg, 'original_temp.jpg', 'Quality', 75);
    recompressedOriginal = imread('original_temp.jpg');
    
    imwrite(forgedImg, 'forged_temp.jpg', 'Quality', 75);
    recompressedForged = imread('forged_temp.jpg');

    elaOriginal = imabsdiff(originalImg, recompressedOriginal);
    elaForged = imabsdiff(forgedImg, recompressedForged);

    elaForgedEnhanced = imadjust(rgb2gray(elaForged));

    figure('Name', 'Error Level Analysis (Enhanced)', 'NumberTitle', 'off');
    subplot(1,2,1); imshow(imadjust(rgb2gray(elaOriginal))); title('Original ELA (Enhanced)');
    subplot(1,2,2); imshow(elaForgedEnhanced); title('Forged ELA (Enhanced)');

    elaMask = elaForgedEnhanced > 30;
    changed = sum(elaMask(:));
    total = numel(elaMask);
    accuracy = 100 * (1 - changed / total);

    fprintf('\n✅ ELA completed. Examine bright regions for tampering clues.\n');
    fprintf('ELA Accuracy Estimate: %.2f%%\n', accuracy);
end

function accuracy = analyzeNoiseVariance(originalGray, forgedGray)
    noise1 = stdfilt(originalGray, ones(3));
    noise2 = stdfilt(forgedGray, ones(3));

    figure('Name', 'Noise Variance Analysis');
    subplot(1,2,1); imshow(mat2gray(noise1)); title('Original Noise Map');
    subplot(1,2,2); imshow(mat2gray(noise2)); title('Forged Noise Map');

    noiseDiff = abs(noise1 - noise2);
    changed = sum(noiseDiff(:) > 10);
    total = numel(noiseDiff);
    accuracy = 100 * (1 - changed / total);

    fprintf('\nNoise Variance Analysis completed.\n');
    fprintf('Noise Variance Accuracy Estimate: %.2f%%\n', accuracy);
end

function accuracy = detectCopyMove(img)
    % Setup VLFeat
    run("C:\Users\ALOK\Downloads\vlfeat-0.9.21\vlfeat-0.9.21\toolbox\vl_setup.m");

    if size(img,3)==3
        grayImg = single(rgb2gray(img));
    else
        grayImg = single(img);
    end

    % Extract SIFT features
    [f, d] = vl_sift(grayImg);
    [matches, ~] = vl_ubcmatch(d, d);

    % Get matched keypoints
    pts1 = f(1:2, matches(1,:))';
    pts2 = f(1:2, matches(2,:))';

    % Remove matches with very small distances (likely self-match)
    distance = sqrt(sum((pts1 - pts2).^2, 2));
    minDist = 10; % Pixels - tweak as needed
    validIdx = distance > minDist;
    pts1 = pts1(validIdx, :);
    pts2 = pts2(validIdx, :);

    % RANSAC: Filter matches using geometric transformation
    if size(pts1, 1) >= 3
        [tform, inlierIdx] = estimateGeometricTransform2D(pts1, pts2, 'similarity', 'MaxDistance', 5);
        pts1 = pts1(inlierIdx, :);
        pts2 = pts2(inlierIdx, :);
    else
        fprintf('\n⚠️ Not enough valid keypoints for RANSAC filtering.\n');
        inlierIdx = true(size(pts1, 1), 1);
    end

    % Display matches
    figure('Name', 'Copy-Move Forgery Detection (Improved)');
    showMatchedFeatures(img, img, pts1, pts2, 'montage');
    title('Filtered Copy-Move Matches');

    % Accuracy estimation
    totalKeypoints = size(f, 2);
    matched = sum(inlierIdx);
    accuracy = 100 * (1 - matched / totalKeypoints);

    fprintf('\n🔍 Copy-Move Detection (Improved):\n');
    fprintf('Total Keypoints: %d, Valid Matches after RANSAC: %d\n', totalKeypoints, matched);
    fprintf('Copy-Move Accuracy Estimate: %.2f%%\n', accuracy);

    % Store accuracy to global for plotting (optional enhancement)
    assignin('base', 'copyMoveAccuracy', accuracy);
end


function accuracy = compareMetadata(imgPath1, imgPath2)
    info1 = imfinfo(imgPath1);
    info2 = imfinfo(imgPath2);

    fprintf('\nMetadata Comparison:\n');
    fields = intersect(fieldnames(info1), fieldnames(info2));
    mismatchCount = 0;

    for i = 1:length(fields)
        val1 = info1.(fields{i});
        val2 = info2.(fields{i});
        if ischar(val1) || isstring(val1)
            isDiff = ~strcmp(string(val1), string(val2));
        else
            isDiff = ~isequal(val1, val2);
        end
        if isDiff
            fprintf('⚠️ %s differs.\n', fields{i});
            mismatchCount = mismatchCount + 1;
        end
    end

    accuracy = 100 * (1 - mismatchCount / length(fields));
    fprintf('✅ Metadata comparison complete.\n');
    fprintf('Metadata Accuracy Estimate: %.2f%%\n', accuracy);
end

function plotAccuracyComparison(pixelAcc, elaAcc, noiseAcc, copyMoveAcc, metadataAcc)
    accuracies = [pixelAcc, elaAcc, noiseAcc, copyMoveAcc, metadataAcc];
    labels = {'Pixel Diff', 'ELA', 'Noise Var', 'Copy-Move', 'Metadata'};

    figure('Name', 'Analysis Accuracy Comparison', 'NumberTitle', 'off');
    bar(accuracies, 'FaceColor', [0.2 0.5 0.9]);
    set(gca, 'XTickLabel', labels, 'XTick', 1:numel(labels), 'FontSize', 12);
    ylabel('Accuracy (%)', 'FontSize', 12);
    title('Forgery Detection Analysis Accuracy', 'FontSize', 14);
    ylim([0 100]);
    grid on;
end
