clc;
clear;
close all;

%% =========================
% BASIC VIDEO LICENSE PLATE DETECTION
% Samples frames and votes on the most likely plate text
% =========================

videoPath = 'car_video.mp4';   % Change this to your video file
frameStep = 5;                 % Process every 5th frame
outputVideoPath = 'annotated_car_video.mp4';

if ~isfile(videoPath)
    error('Video file not found: %s', videoPath);
end

videoReader = VideoReader(videoPath);
videoWriter = VideoWriter(outputVideoPath, 'MPEG-4');
videoWriter.FrameRate = videoReader.FrameRate;
open(videoWriter);

frameIndex = 0;
results = struct('frame', {}, 'text', {}, 'score', {}, 'crop', {});

while hasFrame(videoReader)
    frame = readFrame(videoReader);
    frameIndex = frameIndex + 1;
    outputFrame = frame;

    if mod(frameIndex - 1, frameStep) ~= 0
        writeVideo(videoWriter, outputFrame);
        continue;
    end

    fprintf('Processing frame %d\n', frameIndex);

    try
        [foundPlate, bestText, bestScore, bestCrop, bestBox] = detectPlateInFrame(frame);
        if foundPlate
            resultIndex = numel(results) + 1;
            results(resultIndex).frame = frameIndex;
            results(resultIndex).text = bestText;
            results(resultIndex).score = bestScore;
            results(resultIndex).crop = bestCrop;

            outputFrame = insertShape(outputFrame, 'Rectangle', bestBox, ...
                'Color', 'green', 'LineWidth', 3);
            labelPosition = [bestBox(1), max(1, bestBox(2) - 28)];
            outputFrame = insertText(outputFrame, labelPosition, bestText, ...
                'BoxColor', 'yellow', 'TextColor', 'black', 'FontSize', 18);
        end
    catch ME
        fprintf('Frame %d failed: %s\n', frameIndex, ME.message);
    end

    writeVideo(videoWriter, outputFrame);
end

close(videoWriter);

if isempty(results)
    disp('No valid plate was detected in the video.');
else
    allTexts = {results.text};
    [uniqueTexts, ~, textIds] = unique(allTexts);
    counts = accumarray(textIds(:), 1);
    [~, bestId] = max(counts);
    finalText = uniqueTexts{bestId};

    matchingResults = results(strcmp(allTexts, finalText));
    [~, strongestIdx] = max([matchingResults.score]);
    bestResult = matchingResults(strongestIdx);

    figure('Name', 'Final Video Result', 'Position', [200 200 1000 400]);
    subplot(1,2,1);
    imshow(bestResult.crop);
    title(sprintf('Best Crop (Frame %d)', bestResult.frame));

    subplot(1,2,2);
    axis off;
    text(0.05, 0.65, 'Most Frequent Plate:', 'FontSize', 14, 'FontWeight', 'bold');
    text(0.05, 0.45, finalText, 'FontSize', 24, 'Color', [0 0.35 0.8]);
    text(0.05, 0.18, sprintf('Seen in %d sampled frames', counts(bestId)), 'FontSize', 12);

    disp('==============================');
    disp('Final detected plate text from video:');
    disp(finalText);
    disp(['Annotated video saved to: ' outputVideoPath]);
    disp('==============================');
end


function [foundPlate, bestText, bestScore, bestCrop, bestBoxOriginal] = detectPlateInFrame(I)
    if size(I,3) == 3
        gray = rgb2gray(I);
    else
        gray = I;
    end

    resizeFactor = 1;
    if min(size(gray, 1), size(gray, 2)) < 260
        resizeFactor = 2;
        I = imresize(I, 2);
        gray = imresize(gray, 2);
    end

    grayEq = imadjust(gray);
    grayEq = adapthisteq(grayEq);
    grayEq = medfilt2(grayEq, [3 3]);

    imageHeight = size(grayEq, 1);
    imageWidth = size(grayEq, 2);
    searchMask = false(imageHeight, imageWidth);
    searchMask(round(imageHeight * 0.25):round(imageHeight * 0.95), ...
        round(imageWidth * 0.05):round(imageWidth * 0.95)) = true;

    blackhatImg = imbothat(grayEq, strel('rectangle', [15 40]));
    level = graythresh(blackhatImg);
    BW = imbinarize(blackhatImg, level);
    BW = BW & searchMask;
    BW = imopen(BW, strel('rectangle', [2 2]));
    BW = bwareaopen(BW, 15);

    textBlobBoxes = findTextBlobs(BW, grayEq);
    groupedBoxes = groupTextBlobs(textBlobBoxes, size(grayEq));

    bestScore = -inf;
    bestCrop = [];
    bestText = '';
    bestBoxOriginal = [0 0 0 0];
    foundPlate = false;

    for k = 1:size(groupedBoxes, 1)
        box = groupedBoxes(k, :);
        x = box(1);
        y = box(2);
        w = box(3);
        h = box(4);
        ratio = w / h;
        area = w * h;

        if area < 600 || w < 60 || h < 20 || ratio < 2 || ratio > 8
            continue;
        end

        x1 = max(1, floor(x));
        y1 = max(1, floor(y));
        x2 = min(size(I,2), ceil(x + w - 1));
        y2 = min(size(I,1), ceil(y + h - 1));

        if x2 <= x1 || y2 <= y1
            continue;
        end

        crop = I(y1:y2, x1:x2, :);
        if size(crop,3) == 3
            cropGray = rgb2gray(crop);
        else
            cropGray = crop;
        end

        cropGray = imadjust(cropGray);
        cropGray = adapthisteq(cropGray);
        cropGray = medfilt2(cropGray, [3 3]);
        cropGray = imresize(cropGray, 2);

        cropBW = imbinarize(cropGray, 'adaptive', ...
            'ForegroundPolarity', 'dark', 'Sensitivity', 0.42);
        cropBW = ~cropBW;
        cropBW = bwareaopen(cropBW, 20);

        resultGray = ocr(cropGray, 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789');
        textGray = regexprep(upper(resultGray.Text), '[^A-Z0-9]', '');
        confGray = mean(resultGray.CharacterConfidences, 'omitnan');
        scoreGray = scoreDetectedText(textGray, confGray);

        resultBW = ocr(uint8(cropBW) * 255, 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789');
        textBW = regexprep(upper(resultBW.Text), '[^A-Z0-9]', '');
        confBW = mean(resultBW.CharacterConfidences, 'omitnan');
        scoreBW = scoreDetectedText(textBW, confBW);

        if scoreBW > scoreGray
            cleanText = textBW;
            ocrScore = scoreBW;
        else
            cleanText = textGray;
            ocrScore = scoreGray;
        end

        textLen = length(cleanText);
        if textLen >= 5 && textLen <= 8
            lenScore = 10;
        elseif textLen >= 4 && textLen <= 10
            lenScore = 6;
        elseif textLen >= 2
            lenScore = 2;
        else
            lenScore = 0;
        end

        yCenter = y + h/2;
        if yCenter > size(I,1) * 0.4
            locationScore = 2;
        else
            locationScore = 0;
        end

        brightnessScore = mean(cropGray(:)) / 255;
        groupScore = groupedBoxes(k, 5);
        totalScore = lenScore + groupScore + locationScore + brightnessScore + ocrScore;

        if totalScore > bestScore && textLen >= 2
            bestScore = totalScore;
            bestText = cleanText;
            bestCrop = crop;
            bestBoxOriginal = round([x1, y1, x2 - x1, y2 - y1] / resizeFactor);
            foundPlate = true;
        end
    end
end


function textBlobBoxes = findTextBlobs(BW, grayImage)
    stats = regionprops(BW, 'BoundingBox', 'Area', 'Extent', 'Solidity', 'Eccentricity');
    textBlobBoxes = [];
    imageArea = numel(BW);

    for k = 1:numel(stats)
        box = stats(k).BoundingBox;
        width = box(3);
        height = box(4);
        aspectRatio = width / max(height, 1);
        relativeArea = stats(k).Area / imageArea;

        if relativeArea < 0.00002 || relativeArea > 0.01
            continue;
        end
        if height < 8 || width < 2
            continue;
        end
        if aspectRatio < 0.08 || aspectRatio > 1.2
            continue;
        end
        if stats(k).Extent < 0.15 || stats(k).Extent > 0.95
            continue;
        end
        if stats(k).Solidity < 0.15
            continue;
        end
        if stats(k).Eccentricity > 0.995
            continue;
        end

        x1 = max(1, floor(box(1)));
        y1 = max(1, floor(box(2)));
        x2 = min(size(grayImage, 2), ceil(box(1) + box(3)));
        y2 = min(size(grayImage, 1), ceil(box(2) + box(4)));
        textBlobBoxes = [textBlobBoxes; x1, y1, x2 - x1, y2 - y1]; %#ok<AGROW>
    end
end


function groupedBoxes = groupTextBlobs(textBlobBoxes, imageSize)
    groupedBoxes = [];
    if isempty(textBlobBoxes)
        return;
    end

    centersY = textBlobBoxes(:, 2) + textBlobBoxes(:, 4) / 2;
    heights = textBlobBoxes(:, 4);

    for i = 1:size(textBlobBoxes, 1)
        refCenterY = centersY(i);
        refHeight = heights(i);

        similarRows = abs(centersY - refCenterY) < max(8, 0.6 * refHeight);
        similarHeights = heights > 0.5 * refHeight & heights < 1.8 * refHeight;
        idx = find(similarRows & similarHeights);

        if numel(idx) < 4
            continue;
        end

        selected = textBlobBoxes(idx, :);
        x1 = min(selected(:, 1));
        y1 = min(selected(:, 2));
        x2 = max(selected(:, 1) + selected(:, 3));
        y2 = max(selected(:, 2) + selected(:, 4));
        width = x2 - x1;
        height = y2 - y1;
        aspectRatio = width / max(height, 1);

        if aspectRatio < 2.0 || aspectRatio > 8.5
            continue;
        end
        if width < 30 || height < 10
            continue;
        end

        padX = round(0.10 * width);
        padY = round(0.25 * height);
        x1 = max(1, x1 - padX);
        y1 = max(1, y1 - padY);
        x2 = min(imageSize(2), x2 + padX);
        y2 = min(imageSize(1), y2 + padY);

        rowSpread = std(double(centersY(idx))) / max(imageSize(1), 1);
        alignmentScore = max(0, 1 - 4 * rowSpread);
        groupScore = 2.5 * min(numel(idx), 8) + 8 * alignmentScore;

        groupedBoxes = [groupedBoxes; x1, y1, x2 - x1, y2 - y1, groupScore]; %#ok<AGROW>
    end

    if ~isempty(groupedBoxes)
        groupedBoxes = unique(round(groupedBoxes), 'rows', 'stable');
        groupedBoxes = sortrows(groupedBoxes, 5, 'descend');
        groupedBoxes = groupedBoxes(1:min(10, size(groupedBoxes, 1)), :);
    end
end


function textScore = scoreDetectedText(cleanText, confidence)
    if isnan(confidence)
        confidence = 0;
    end

    textLen = length(cleanText);
    hasLetter = ~isempty(regexp(cleanText, '[A-Z]', 'once'));
    hasDigit = ~isempty(regexp(cleanText, '[0-9]', 'once'));

    textScore = 0;
    textScore = textScore + 0.08 * min(textLen, 10);
    textScore = textScore + 0.02 * max(confidence, 0);
    textScore = textScore + 0.2 * double(hasLetter);
    textScore = textScore + 0.2 * double(hasDigit);
    if textLen >= 5 && textLen <= 8
        textScore = textScore + 0.5;
    end
end

