myFolder = './data/CNN_static_data/Train';
if ~isfolder(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.png');
pngFiles = dir(filePattern);
for k = 1:length(pngFiles)
  baseFileName = pngFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imageArray = imread(fullFileName);
  imshow(imageArray);  % Display image.
  drawnow; % Force display to update immediately.
end


F=imrotate(imresize(cdata,[512, 512]), 45, 'bicubic', 'crop');
figure(1);imshow(F,[])
F=fft2(F);
Fsh=fftshift(F);
% imshow(abs(Fsh), [])
figure(2); imshow(log(1+abs(Fsh)),[])
% Fsh = imrotate(Fsh,10, 'bicubic', 'crop');
% Fsh = Fsh(65:448,65:448);
Fsh=Fsh .* gaussianFilter(512,2,40);
figure(3); imshow(log(1+abs(Fsh)),[])
figure(4); imshow(imresize(abs(ifft2(ifftshift(Fsh))),[64,64]),[])
