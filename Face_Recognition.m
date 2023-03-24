w=load_database();
%% Initializations
% We randomly pick an image from our database and use the rest of the
% images for training. Training is done on 49 pictues. We later
% use the randomly selectted picture to test the algorithm.
[rows, columns] = size(w);
ri = randperm(columns, 1);           % Randomly pick an index.
r=w(:,ri);                              % r contains the image we later on will use to test the algorithm
allColumns = 1 : columns;
otherColumns = setdiff(allColumns, ri);     % Get a list of all columns other than ri.
v = w(:, otherColumns);                 % v contains the rest of the 49 images.
N=20;                               % Number of signatures used for each image.
%% Subtracting the mean from v
image=uint8(ones(1,size(v,2)));         %white
mean_val=uint8(mean(v,2));                 % mean_val of all images.
mean_removed=v-uint8(single(mean_val)*single(image));   % mean_removed is v with the mean removed. 
%% Calculating eignevectors of the correlation matrix
% We are picking N of the eigenfaces.
L=single(mean_removed)'*single(mean_removed);
[V,D]=eig(L);                       %calculates the eigenvector for images
V=single(mean_removed)*V;
V=V(:,end:-1:end-(N-1));            % Pick the eignevectors corresponding to the 10 largest eigenvalues. 
%% Calculating the signature for each image
all_img_signatures=zeros(size(v,2),N);      
for i=1:size(v,2);
    all_img_signatures(i,:)=single(mean_removed(:,i))'*V;    % each row contains the signature of individual images
end
%% Recognition 
%  Now, we run the algorithm and see if we can correctly recognize the face. 
subplot(121); 
imshow(reshape(r,112,92));
title('Searching for: ','FontWeight','bold','Fontsize',16,'color','red');
p=r-mean_val;                       %  random - mean
s=single(p)'*V;                     % transpose & multiply
z=[];
for i=1:size(v,2)
    z=[z,norm(all_img_signatures(i,:)-s,2)];            %1:1 mapping with signatures
    drawnow;
end
[a,i]=min(z);
subplot(122);
imshow(reshape(v(:,i),112,92));
title('Found!','FontWeight','bold','Fontsize',16,'color','red');