clc;
clear;
close all;

%% =========================
% LICENSE PLATE DETECTION
% Text-first candidate grouping
% =========================

imageFolder = 'imagesToDetect';
supportedExtensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};

if ~isfolder(imageFolder)
    error('Folder not found: %s', imageFolder);
end

imageFiles = [];
for i = 1:numel(supportedExtensions)
    imageFiles = [imageFiles; dir(fullfile(imageFolder, supportedExtensions{i}))]; %#ok<AGROW>
end

if isempty(imageFiles)
    error('No image files were found inside folder: %s', imageFolder);
end

for imageIndex = 1:numel(imageFiles)
    imagePath = fullfile(imageFiles(imageIndex).folder, imageFiles(imageIndex).name);
    fprintf('\nProcessing image %d of %d: %s\n', imageIndex, numel(imageFiles), imageFiles(imageIndex).name);

    I = imread(imagePath);
    if size(I,3) == 3
        gray = rgb2gray(I);
    else
        gray = I;
    end

    if min(size(gray, 1), size(gray, 2)) < 260
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
    bestBox = [0 0 0 0];
    bestCrop = [];
    bestText = '';
    bestPlateGray = [];
    foundPlate = false;

    % Uncomment this whole figure block when you want to inspect the pipeline.
    % figure('Name',['Plate Detection Pipeline - ' imageFiles(imageIndex).name], 'Position',[100 100 1400 800]);
    %
    % subplot(2,4,1);
    % imshow(I);
    % title('Original Image');
    %
    % subplot(2,4,2);
    % imshow(gray);
    % title('Grayscale');
    %
    % subplot(2,4,3);
    % imshow(grayEq);
    % title('Enhanced');
    %
    % subplot(2,4,4);
    % imshow(blackhatImg, []);
    % title('Blackhat');
    %
    % subplot(2,4,5);
    % imshow(BW);
    % title('Text Blob Mask');
    %
    % subplot(2,4,6);
    % imshow(I);
    % title('Grouped Text Candidates');
    % hold on;

    for k = 1:size(groupedBoxes, 1)
        box = groupedBoxes(k, :);
        % rectangle('Position', box(1:4), 'EdgeColor', 'y', 'LineWidth', 1.5);

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

        try
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
                textLen = length(cleanText);
                ocrScore = scoreBW;
                bestOcrPreview = uint8(cropBW) * 255;
            else
                cleanText = textGray;
                textLen = length(cleanText);
                ocrScore = scoreGray;
                bestOcrPreview = cropGray;
            end

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

            % disp(['Candidate ', num2str(k), ...
            %       ' | Text="', cleanText, '"', ...
            %       ' | Len=', num2str(textLen), ...
            %       ' | Ratio=', num2str(ratio), ...
            %       ' | Score=', num2str(totalScore)]);

            if totalScore > bestScore && textLen >= 2
                bestScore = totalScore;
                bestBox = box(1:4);
                bestCrop = crop;
                bestText = cleanText;
                bestPlateGray = bestOcrPreview;
                foundPlate = true;
            end
        catch ME
            % disp(['Candidate ', num2str(k), ' OCR failed: ', ME.message]);
        end
    end
    % hold off;

    % subplot(2,4,7);
    % imshow(I);
    % title('Best Plate Detection');
    % hold on;
    % if foundPlate
    %     rectangle('Position', bestBox, 'EdgeColor', 'g', 'LineWidth', 2.5);
    % end
    % hold off;
    %
    % subplot(2,4,8);
    % if foundPlate
    %     imshow(bestCrop);
    %     title(['Best Crop: ', bestText]);
    % else
    %     imshow(zeros(100));
    %     title('No Plate Found');
    % end

    if foundPlate
        figure('Name',['Final Result - ' imageFiles(imageIndex).name], 'Position',[200 200 1000 400]);
        subplot(1,2,1);
        imshow(bestCrop);
        title('Detected Plate');

        subplot(1,2,2);
        axis off;
        text(0.05, 0.60, 'Detected Text:', 'FontSize', 14, 'FontWeight', 'bold');
        text(0.05, 0.40, bestText, 'FontSize', 22, 'Color', [0 0.35 0.8]);

        disp('==============================');
        disp(['Final detected plate text for ' imageFiles(imageIndex).name ':']);
        disp(bestText);
        disp('==============================');
    else
        disp(['No valid plate candidate found for ' imageFiles(imageIndex).name '.']);
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
