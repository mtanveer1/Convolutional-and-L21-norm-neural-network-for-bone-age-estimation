function beta=optimize_beta2(X,trainY_temp,option)
[N,L]=size(X);
C = option.C;
D=eye(L); D1=eye(N);
num_iter=option.num_iter;
if strcmp(option.reg,'L21_Reg_Both')
    if size(X,2)<N
        beta = (eye(size(X,2))/C+X'*X) \ X'*trainY_temp;
        while num_iter>0
            D = sqrt(sum(beta.*beta,2)+eps);
            D = 0.5./D;
            D=diag(D);

            xi=trainY_temp-X*beta;
            D1 = sqrt(sum(xi.*xi,2)+eps);
            D1 = 0.5./D1;
            D1=diag(D1);
            temp=X'*D1;
            beta = (D/C+temp*X) \ temp*trainY_temp;
            num_iter=num_iter-1;
        end
    else
        beta = X'*((eye(size(X,1))/C+X*X') \ trainY_temp);
        while num_iter>0
            D = 2*( sqrt(sum(beta.*beta,2)+eps) );
            invD=D;
            D=diag(invD);

            xi=trainY_temp-X*beta;
            D1 = 2*( sqrt(sum(xi.*xi,2)+eps) );
            invD1=D1;
            D1=diag(invD1);
            temp=D*X'; 
            beta = temp*( ( (1/C)*D1+X*temp) \ trainY_temp );
            num_iter=num_iter-1;
        end
    end
else
    disp('Give Proper Regularization')
end
end
