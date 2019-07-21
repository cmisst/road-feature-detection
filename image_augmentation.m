%% Folder info
myFolder = './data/CNN_static_data/Train';
if ~isfolder(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.png');
pngFiles = dir(filePattern);





%% Parallel (file-wise) loop
for k = 1:1%length(pngFiles)
    baseFileName = pngFiles(k).name;
    assert(length(baseFileName)==17)
    fullFileName = fullfile(myFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    imageArray = imresize(imread(fullFileName),[512,512]);
  
    %% Gaussian filter power-rate pair
    power_grid = 5;
    rate_grid = 7;    
    power = logspace(log10(0.3), log10(3), power_grid);
    power_rate_pair = 1.1.*(logspace(-0.3,2, rate_grid)' * sqrt(pi) ...
        * gamma(1./power+1)/gamma(1/2+1./power)).^(power/2);
    
    FT=fftshift(fft2(imageArray));
    for i=1:power_grid
        for j=1:rate_grid
            gF=gaussianFilter(512,power(i),power_rate_pair(j,i));
            FT_reconstruct=abs(ifft2(ifftshift(FT.*gF)));
            imshow(FT_reconstruct, [])
        end
    end
  
%   imshow(imageArray);  % Display image.
%   drawnow; % Force display to update immediately.
end
