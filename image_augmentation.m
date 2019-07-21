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
parfor file = 1:length(pngFiles)
    baseFileName = pngFiles(file).name;
    assert(length(baseFileName)==17)
    fullFileName = fullfile(myFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    imageArray = imresize(imread(fullFileName),[64,64]);
  
    %% Gaussian filter power-rate pair
    power_grid = 4;
    rate_grid = 4;    
    power = logspace(log10(0.4), log10(4), power_grid);
    power_rate_pair = (logspace(0,0.9, rate_grid)' * sqrt(pi) ...
        * gamma(1./power+1)/gamma(1/2+1./power)).^(power/2);
    
    for t=0:3:719
        if(t==360)
            imageArray=imageArray';
        end
        FT=fftshift(fft2(imrotate(imageArray, t, 'bicubic', 'crop')));

        for i=1:size(power_rate_pair,2)
            for j=1:size(power_rate_pair,1)
                tic
                gF=gaussianFilter(64, power(i), power_rate_pair(j,i));
                for r=0:6:89
                    % imshow(imrotate(gF, r, 'bicubic', 'crop'), [])
                    FT_reconstruct=abs(ifft2(ifftshift(...
                    FT .* imrotate(gF, r, 'bicubic', 'crop'))));
                    % imshow(imresize(FT_reconstruct,[256,256], 'nearest'), [])

                    % PCA part
                    svd_count=[64:-4:36, 32:-2:18, 16:-1:13];
                    for p = 1:length(svd_count)
                        PCA_reconstruct = pca_reconstruct(FT_reconstruct, svd_count(p));
                        name = filename_generate('./data/CNN_static_data/Train_augmented/',...
                            baseFileName, t, i, j, r, svd_count(p));
                        imwrite(uint8(PCA_reconstruct), name)
                        % imshow(imresize(PCA_reconstruct,[256,256], 'nearest'), [])
                    end
                end
                toc
            end
        end
    end
    
end
