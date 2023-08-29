function GaussLabel= GaussLabel_func(label,option)
num_class=240; 
try
    sigma=option.sigma;
catch
    sigma=2^8;
end
x=1:240;

temp=-(x-repmat(label,1,num_class)).^2;
temp=exp((1/(2*sigma.^2)*temp));
GaussLabel=temp./(sigma*2*pi);
end
