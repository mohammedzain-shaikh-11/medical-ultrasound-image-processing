function [bmodeImage, envelopeImage] = ...
    reconstructUltrasoundDAS_GPU_exact( ...
    rfData, params, depthAxis, lateralAxis)

% GPU DAS using an exact CPU-generated sample-index map.
% The same RF sample indices as the CPU reference are used.

c = params.c;
fs = params.fs;
numElements = params.numElements;
elementSpacing = params.elementSpacing;

numDepth = numel(depthAxis);
numLateral = numel(lateralAxis);

% Element positions
elementPositions = ...
    ((0:numElements-1) - ...
    (numElements-1)/2) * elementSpacing;

% Image grid on CPU
[X,Z] = meshgrid(lateralAxis,depthAxis);

% Precompute exact CPU sample indices
sampleIndexMap = zeros(numDepth,numLateral,numElements,'uint32');
validMap = false(numDepth,numLateral,numElements);

for elem = 1:numElements

    distance = sqrt( ...
        (X-elementPositions(elem)).^2 + Z.^2);

    delay = 2*distance/c;

    sampleIndex = round(delay*fs)+1;

    valid = sampleIndex >= 1 & ...
        sampleIndex <= size(rfData,1);

    sampleIndex(~valid) = 1;

    sampleIndexMap(:,:,elem) = uint32(sampleIndex);
    validMap(:,:,elem) = valid;

end

% Move RF data to GPU
rfGPU = gpuArray(single(rfData));

% GPU accumulation
envelopeGPU = gpuArray.zeros( ...
    numDepth,numLateral,'single');

for elem = 1:numElements

    % Exact CPU sample-index map
    indices = sampleIndexMap(:,:,elem);

    % Convert this image-shaped index map to a vector
    indicesVector = indices(:);

    % RF samples
    rfColumn = rfGPU(:,elem);

    contributionVector = rfColumn(indicesVector);

    % Restore 256 x 256 image shape
    contribution = reshape( ...
        contributionVector,numDepth,numLateral);

    % Remove invalid samples
    valid = validMap(:,:,elem);

    contribution(~valid) = 0;

    envelopeGPU = envelopeGPU + contribution;

end

% Envelope detection
analyticSignal = hilbert(envelopeGPU);
envelopeGPU = abs(analyticSignal);

% Normalize
maxValue = max(envelopeGPU(:));

if maxValue > 0
    envelopeGPU = envelopeGPU ./ maxValue;
end

% Convert to dB
bmodeGPU = 20*log10(envelopeGPU + eps('single'));

bmodeGPU(bmodeGPU < -40) = -40;

% Return results to CPU
bmodeImage = gather(bmodeGPU);
envelopeImage = gather(envelopeGPU);

end