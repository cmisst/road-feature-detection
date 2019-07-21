function [PCA_reconstruct] = pca_reconstruct(imageMtx, svd_count)
    [U, s, V]=svd(imageMtx);
    s=diag(s);
    s(svd_count:end)=0;
    PCA_reconstruct=U*diag(s)*V';
end



% F=imrotate(imresize(imageArray,[512, 512]), 45, 'bicubic', 'crop');
% % figure(1);imshow(F,[])
% F=fft2(F);
% Fsh=fftshift(F);
% % imshow(abs(Fsh), [])
% figure(2); imshow(log(1+abs(Fsh)),[])
% % Fsh = imrotate(Fsh,10, 'bicubic', 'crop');
% % Fsh = Fsh(65:448,65:448);
% Fsh=Fsh .* gF;
% figure(3); imshow(log(1+abs(Fsh)),[])
% figure(4); imshow(imresize(abs(ifft2(ifftshift(Fsh))),[64,64]),[])
