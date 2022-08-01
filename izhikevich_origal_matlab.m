% spnet.m: Spiking network with axonal conduction delays and STDP % Created by Eugene M.Izhikevich. February 3, 2004
M=100;
D=20;
% excitatory neurons 
Ne=800; 
Ni=200; 
N=Ne+Ni;
a=[0.02*ones(Ne,1); 0.1*ones(Ni,1)];
d=[8*ones(Ne,1);2*ones(Ni,1)]; 
sm=10; % maximal synaptic strength
post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)]);
s=[6*ones(Ne,M);-5*ones(Ni,M)]; 
sd=zeros(N,M);
do_plot = 0;
for i=1:N
    if i<=Ne 
        for j=1:D
            delays{i,j}=M/D*(j-1)+(1:M/D); 
        end
    else 
        delays{i,1}=1:M;
    end
    pre{i}=find(post==i&s>0); 
    aux{i}=N*(D-1-ceil(ceil(pre{i}/N)/(M/D)))+1+mod(pre{i}-1,N);
end
STDP = zeros(N,1001+D); 
v = -65*ones(N,1);
u = 0.2.*v; 
firings=[-D 0];

for sec=1:60*60*24 
    tic
    for t=1:1000
        I=zeros(N,1); I(ceil(N*rand))=20;
        fired = find(v>=30); 
        v(fired)=-65; 
        u(fired)=u(fired)+d(fired); 
        STDP(fired,t+D)=0.1;
        for k=1:length(fired)
            sd(pre{fired(k)})=sd(pre{fired(k)})+STDP(N*t+aux{fired(k)}); 
        end
        firings=[firings;t*ones(length(fired),1),fired]; 
        k=size(firings,1);
        while firings(k,1)>t-D
            del=delays{firings(k,2),t-firings(k,1)+1};
            ind = post(firings(k,2),del);
            I(ind)=I(ind)+s(firings(k,2), del)'; 
            sd(firings(k,2),del)=sd(firings(k,2),del)-1.2*STDP(ind,t+D)'; 
            k=k-1;
        end
        v=v+0.5*((0.04*v+5).*v+140-u+I); 
        v=v+0.5*((0.04*v+5).*v+140-u+I); 
        u=u+a.*(0.2*v-u); 
        STDP(:,t+D+1)=0.95*STDP(:,t+D);
        % for numerical % stability time %stepis0.5ms %tau=20ms
    end
    toc
    if do_plot ==1
        plot(firings(:,1),firings(:,2),'.');
        axis([0 1000 0 N]); 
        drawnow; 
    end

    STDP(:,1:D+1)=STDP(:,1001:1001+D);
    ind = find(firings(:,1) > 1001-D);
    firings=[-D 0;firings(ind,1)-1000,firings(ind,2)]; 
    s(1:Ne,:)=max(0,min(sm,0.01+s(1:Ne,:)+sd(1:Ne,:))); 
    sd=0.9*sd;
end