% =========================================================================
% -- Simulator for 1-bit Massive MU-MIMO Precoding in VLSI with CxPO
% -------------------------------------------------------------------------
% -- (c) 2016 Christoph Studer, Oscar Castañeda, and Sven Jacobsson
% -- e-mail: studer@cornell.edu, oc66@cornell.edu, and
% -- sven.jacobsson@ericsson.com (version 0.1; August 14, 2017)
% -------------------------------------------------------------------------
% -- If you use this simulator or parts of it, then you must cite our
% -- journal paper:
% --   Oscar Castañeda, Sven Jacobsson, Giuseppe Durisi, Mikael Coldrey,
% --   Tom Goldstein, and Christoph Studer, "1-bit Massive MU-MIMO
% --   Precoding in VLSI," IEEE Journal on Emerging and Selected Topics in
% --   Circuits and Systems (JETCAS), to appear in 2017
% -- and clearly mention this in your paper
% -------------------------------------------------------------------------
% -- REMEMBER: C1PO + C2PO = C(1+2)PO = C3PO :)
% =========================================================================

function precoder_sim(varargin)

% -- set up default/custom parameters

if isempty(varargin)
    
    disp('using default simulation settings and parameters...')
    
    % set default simulation parameters
    par.runId = 0;       % simulation ID (used to reproduce results)
    par.L = 2;           % number of DAC levels per I or Q dimension (must be 2!!!)
    par.U = 16;          % number of single-antenna users
    par.B = 256;         % number of base-station antennas (B>>U)
    par.mod = '16QAM';   % modulation type: 'BPSK','QPSK','16QAM','64QAM','8PSK'
    par.trials = 1e3;    % number of Monte-Carlo trials (transmissions)
    par.NTPdB_list = ... % list of normalized transmit power [dB] values
        -10:2:20;        % to be simulated
    par.precoder = ...   % precoding scheme(s) to be evaluated
        {'ZF','MRT','ZFQ','MRTQ','SQUID','C1PO','C2PO'};
    par.save = true;     % save results (true,false)
    par.plot = true;     % plot results (true,false)
    
    % *** SQUID specific
    %
    % note that the SQUID code includes two more algorithm parameters that
    % must be tuned for best performance (if you know what you are doing).
    par.SQUID.iterations = 200;
    
    % *** C1PO specific
    %
    % reasonable parameters for C1PO with different system configurations
    % please optimize manually for best performance (depends on # of iters)
    %
    % BxU    | mod.  | gamma | delta | rho
    % -------+-------+-------+-------+------
    % 32x16  | BPSK  | 2^5   | 6.4   | 1.25
    % 64x16  | BPSK  | 2^4   | 3.2   | 1.25
    % 128x16 | BPSK  | 2^2   | 0.8   | 1.25
    % 256x16 | BPSK  | 2^3   | 1.6   | 1.25
    % -------+-------+-------+-------+------
    % 32x16  | QPSK  | 2^5   | 6.4   | 1.25
    % 64x16  | QPSK  | 2^4   | 3.2   | 1.25
    % 128x16 | QPSK  | 2^2   | 0.8   | 1.25
    % 256x16 | QPSK  | 2^3   | 1.6   | 1.25
    % -------+-------+-------+-------+-------
    % 256x16 | 16QAM | 2^1   | 0.4   | 1.25
    % -------+-------+-------+-------+-------
    % 256x16 | 64QAM | 14    | 2.8   | 1.25
    
    par.C1PO.gamma = 2^1; % good for 256x16 with 16-QAM
    par.C1PO.rho = 1.25; % rho = gamma/(gamma-delta) [aka. pushfactor]
    par.C1PO.iterations = 25; % max number of iterations
    
    % *** C2PO specific
    %
    % reasonable parameters for C2PO with different system configurations
    % please optimize manually for best performance (depends on # of iters)
    %
    % BxU    | mod.  | tau   | delta | rho
    % -------+-------+-------+-------+------
    % 32x16  | BPSK  | 2^-6  | 12.8  | 1.25
    % 64x16  | BPSK  | 2^-7  | 25.6  | 1.25
    % 128x16 | BPSK  | 2^-7  | 25.6  | 1.25
    % 256x16 | BPSK  | 2^-8  | 51.2  | 1.25
    % -------+-------+-------+-------+------
    % 32x16  | QPSK  | 2^-6  | 12.8  | 1.25
    % 64x16  | QPSK  | 2^-7  | 25.6  | 1.25
    % 128x16 | QPSK  | 2^-7  | 25.6  | 1.25
    % 256x16 | QPSK  | 2^-8  | 51.2  | 1.25
    % -------+-------+-------+-------+-------
    % 256x16 | 16QAM | 2^-8  | 51.2  | 1.25
    % -------+-------+-------+-------+-------
    % 256x16 | 64QAM | 2^-8  | 51.2  | 1.25
    
    par.C2PO.tau = 2^(-8); % good for 256x16 with 16-QAM
    par.C2PO.rho = 1.25; % rho = 1/(1-tau*delta) [aka. pushfactor]
    par.C2PO.iterations = 25; % max number of iterations
    
else
    
    disp('use custom simulation settings and parameters...')
    par = varargin{1};   % only argument is par structure
    
end

% -- initialization

% the methods have only been checked for 1-bit transmission
% an extension to multi-bit needs more work :)
if par.L~=2
    error('This simulator is specifically designed for 1-bit scenarios')
end

% use runId random seed (enables reproducibility)
rng(par.runId);

% simulation name (used for saving results)
par.simName = ['ERR_',num2str(par.U),'x',num2str(par.B), '_', ...
    par.mod, '_', num2str(par.trials),'Trials'];

% set up Gray-mapped constellation alphabet (according to IEEE 802.11)
switch (par.mod)
    case 'BPSK',
        par.symbols = [ -1 1 ];
    case 'QPSK',
        par.symbols = [ -1-1i,-1+1i,+1-1i,+1+1i ];
    case '16QAM',
        par.symbols = [ -3-3i,-3-1i,-3+3i,-3+1i, ...
            -1-3i,-1-1i,-1+3i,-1+1i, ...
            +3-3i,+3-1i,+3+3i,+3+1i, ...
            +1-3i,+1-1i,+1+3i,+1+1i ];
    case '64QAM',
        par.symbols = [ -7-7i,-7-5i,-7-1i,-7-3i,-7+7i,-7+5i,-7+1i,-7+3i, ...
            -5-7i,-5-5i,-5-1i,-5-3i,-5+7i,-5+5i,-5+1i,-5+3i, ...
            -1-7i,-1-5i,-1-1i,-1-3i,-1+7i,-1+5i,-1+1i,-1+3i, ...
            -3-7i,-3-5i,-3-1i,-3-3i,-3+7i,-3+5i,-3+1i,-3+3i, ...
            +7-7i,+7-5i,+7-1i,+7-3i,+7+7i,+7+5i,+7+1i,+7+3i, ...
            +5-7i,+5-5i,+5-1i,+5-3i,+5+7i,+5+5i,+5+1i,+5+3i, ...
            +1-7i,+1-5i,+1-1i,+1-3i,+1+7i,+1+5i,+1+1i,+1+3i, ...
            +3-7i,+3-5i,+3-1i,+3-3i,+3+7i,+3+5i,+3+1i,+3+3i ];
    case '8PSK',
        par.symbols = [ exp(1i*2*pi/8*0), exp(1i*2*pi/8*1), ...
            exp(1i*2*pi/8*7), exp(1i*2*pi/8*6), ...
            exp(1i*2*pi/8*3), exp(1i*2*pi/8*2), ...
            exp(1i*2*pi/8*4), exp(1i*2*pi/8*5) ];
end

% compute symbol energy
par.Es = mean(abs(par.symbols).^2);

% - quantizer paremeters
% optimal LSB for 2 < L < 16 quantization levels
lsb_list = [ 1.59628628628629,  ...
    1.22515515515516,  ...
    0.994694694694695, ...
    0.842052052052052, ...
    0.734304304304304, ...
    0.650500500500501, ...
    0.584654654654655, ...
    0.533773773773774, ...
    0.491871871871872, ...
    0.455955955955956, ...
    0.423033033033033, ...
    0.396096096096096, ...
    0.375145145145145, ...
    0.354194194194194, ...
    0.336236236236236 ];
% resolution (number of bits) of the DACs
par.Q = log2(par.L);
% least significant bit
par.lsb = lsb_list(par.L-1)/sqrt(2*par.B);
% clip level
par.clip = par.lsb*par.L/2;
% quantizer labels and thresholds
[~, ~, par.labels, par.thresholds, ~] = uniquantiz(1, par.lsb, par.L);
% normalization constant
par.alpha = sqrt( 1/(2*par.B) ...
    /sum(par.labels.^2.*( ...
    normcdf(par.thresholds(2:end)*sqrt(2*par.B)) ...
    -normcdf(par.thresholds(1:end-1)*sqrt(2*par.B)))));
% scale quantization labels
par.labels = par.alpha*par.labels;
% quantizer alphabet
par.alphabet = combvec(par.labels, par.labels);
par.alphabet = par.alphabet(1,:) + 1i*par.alphabet(2,:);
% quantizer-mapping function
par.quantizer = @(x) par.alpha * uniquantiz(x, par.lsb, par.L);
% equivalent (average) quantizer gain
par.F = par.alpha*par.lsb*...
    sum(normpdf(par.thresholds(2:end-1),0,1/sqrt(2*par.B)));

% precompute bit labels
par.bps = log2(length(par.symbols)); % number of bits per symbol
par.bits = de2bi(0:length(par.symbols)-1,par.bps,'left-msb');

% track simulation time
time_elapsed = 0;

% -- start simulation

% - initialize result arrays (detector x normalized transmit power)
% vector error rate
res.VER = zeros(length(par.precoder),length(par.NTPdB_list));
% symbol error rate
res.SER = zeros(length(par.precoder),length(par.NTPdB_list));
% bit error rate
res.BER = zeros(length(par.precoder),length(par.NTPdB_list));
% error-vector magnitude
res.EVM = zeros(length(par.precoder),length(par.NTPdB_list));
% SINDR
res.SINDR = zeros(length(par.precoder),length(par.NTPdB_list));
% transmit power
res.TxPower = zeros(length(par.precoder),length(par.NTPdB_list));
% receive power
res.RxPower = zeros(length(par.precoder),length(par.NTPdB_list));
% simulation beamforming time
res.TIME = zeros(length(par.precoder),length(par.NTPdB_list));

% compute noise variances to be considered
N0_list = 10.^(-par.NTPdB_list/10);

% generate random bit stream (antenna x bit x trial)
bits = randi([0 1],par.U,par.bps,par.trials);

% trials loop
tic
for t=1:par.trials
    
    % generate transmit symbol
    idx = bi2de(bits(:,:,t),'left-msb')+1;
    s = par.symbols(idx).';
    
    % generate iid Gaussian channel matrix and noise vector
    n = sqrt(0.5)*(randn(par.U,1)+1i*randn(par.U,1));
    H = sqrt(0.5)*(randn(par.U,par.B)+1i*randn(par.U,par.B));
    
    % algorithm loop
    for d=1:length(par.precoder)
        
        % normalized transmit power loop
        for k=1:length(par.NTPdB_list)
            
            % set noise variance
            N0 = N0_list(k);
            
            % record time used by the beamformer
            starttime = toc;
            
            % beamformers
            switch (par.precoder{d})
                % noise-independent
                case 'ZF',      % ZF beamforming (infinite precision)
                    [x, beta] = ZF(par, s, H);
                case 'ZFQ',     % ZF beamforming (quantized)
                    [x, beta] = ZF(par, s, H);
                    x = par.quantizer(x);
                    beta = beta/par.F;
                case 'MRT',      % MRT beamforming (infinite precision)
                    [x, beta] = MRT(par, s, H);
                case 'MRTQ',     % MRT beamforming (quantized)
                    [x, beta] = MRT(par, s, H);
                    x = par.quantizer(x);
                    beta = beta/par.F;
                case 'C1PO',      % C1PO: biConvex 1-bit PrecOding
                    [x, beta] = C1PO(par, s, H);
                case 'C2PO',      % C2PO: C1PO with simpler preprocessing
                    [x, beta] = C2PO(par, s, H);
                    % noise-dependent
                case 'SQUID',   % SQUID: Squared inifinity-norm relaxation with
                    % Douglas-Rachford splitting
                    [x, beta] = SQUID(par,s,H,N0);
                otherwise,
                    error('par.precoder not specified')
            end
            
            % record beamforming simulation time
            res.TIME(d,k) = res.TIME(d,k) + (toc-starttime);
            
            % transmit data over noisy channel
            Hx = H*x;
            y = Hx + sqrt(N0)*n;
            
            % extract transmit and receive power
            res.TxPower(d,k) = res.TxPower(d) + mean(sum(abs(x).^2));
            res.RxPower(d,k) = res.RxPower(d) + mean(sum(abs(Hx).^2))/par.U;
            
            % user terminals can estimate the beamforming factor beta
            shat = beta*y;
            
            % perform user-side detection
            [~,idxhat] = min(abs(shat*ones(1,length(par.symbols)) ...
                -ones(par.U,1)*par.symbols).^2,[],2);
            bithat = par.bits(idxhat,:);
            
            % -- compute error and complexity metrics
            err = (idx~=idxhat);
            res.VER(d,k) = res.VER(d,k) + any(err);
            res.SER(d,k) = res.SER(d,k) + sum(err)/par.U;
            res.BER(d,k) = res.BER(d,k) + ...
                sum(sum(bits(:,:,t)~=bithat))/(par.U*par.bps);
            res.EVM(d,k) = res.EVM(d,k) + 100*norm(shat - s)^2/norm(s)^2;
            res.SINDR(d,k) = res.SINDR(d,k) + norm(s)^2/norm(shat - s)^2;
            
        end % NTP loop
        
    end % algorithm loop
    
    % keep track of simulation time
    if toc>10
        time=toc;
        time_elapsed = time_elapsed + time;
        fprintf('estimated remaining simulation time: %3.0f min.\n',...
            time_elapsed*(par.trials/t-1)/60);
        tic
    end
    
end % trials loop

% normalize results
res.VER = res.VER/par.trials;
res.SER = res.SER/par.trials;
res.BER = res.BER/par.trials;
res.EVM = res.EVM/par.trials;
res.SINDR = res.SINDR/par.trials;
res.TxPower = res.TxPower/par.trials;
res.RxPower = res.RxPower/par.trials;
res.TIME = res.TIME/par.trials;
res.time_elapsed = time_elapsed;

% -- save final results (par and res structures)

if par.save
    save([ par.simName '_' num2str(par.runId) ],'par','res');
end

% -- show results (generates fairly nice Matlab plots)

if par.plot
    
    % - BER results
    marker_style = {'k-','b:','r--','y-.','g-.','bs--','mv--'};
    figure(1)
    for d=1:length(par.precoder)
        semilogy(par.NTPdB_list,res.BER(d,:),marker_style{d},'LineWidth',2);
        if (d==1)
            hold on
        end
    end
    hold off
    grid on
    box on
    xlabel('normalized transmit power [dB]','FontSize',12)
    ylabel('uncoded bit error rate (BER)','FontSize',12);
    if length(par.NTPdB_list) > 1
        axis([min(par.NTPdB_list) max(par.NTPdB_list) 1e-3 1]);
    end
    legend(par.precoder,'FontSize',12,'location','southwest')
    set(gca,'FontSize',12);
    
end

end

%% Uniform quantizer
function [v, q, vl, vt, c] = uniquantiz(y, lsb, L)

% set clip level
c = lsb*L/2;

% clip signal
if isreal(y)
    yc = max(min(y,c-lsb/1e5),-(c-lsb/1e5));
else
    yc = max(min(real(y),c-lsb/1e5),-(c-lsb/1e5)) ...
        + 1i*max(min(imag(y),c-lsb/1e5),-(c-lsb/1e5));
end

% quantizer
if mod(L,2) == 0
    % midrise quantizer (without clipping)
    Q = @(x) lsb*floor(x/lsb) + lsb/2;
else
    % midtread quantizer (without clipping)
    Q = @(x) lsb*floor(x/lsb + 1/2);
end

% quantize signal
if isreal(y)
    v = Q(yc);
else
    v = Q(real(yc)) + 1i*Q(imag(yc));
end

% quantization error
q = v - y;

% uniform quantization labels
vl = lsb *((0:L-1) - (L-1)/2);

% uniform quantization thresholds
vt = [-realmax*ones(length(lsb),1), ...
    bsxfun(@minus, vl(:,2:end), lsb/2), ...
    realmax*ones(length(lsb),1)];

end

%% Zero-forcing beamforming (with infinite precision)
function [x, beta] = ZF(par, s, H)

% normalization constant (average gain)
rho = sqrt((par.B-par.U)/(par.Es*par.U));

% transmitted signal
x = rho*H'/(H*H')*s;

% beamforming factor
beta = 1/rho;

end

%% Maximum ratio transmission (MRT) beamforming (with infinite precision)
function [x, beta, P] = MRT(par, s, H)

% normalization constant
gmrt = 1/sqrt(par.Es*par.U*par.B); % average gain
% gmrt = 1/sqrt(par.Es*trace(H*H')); % instant gain

% precoding matrix
P = gmrt*H';

% transmitted signal
x = P*s;

% scaling factor
beta = sqrt(par.U*par.Es/par.B);

end

%% C1PO: biConvex 1-bit PrecOding (Algorithm 1)
function [x, beta] = C1PO(par,s,H)

% initial guess
x = H'*s;

% preprocessing with exact inverse
gammainv = 1/par.C1PO.gamma;
Ainv = inv(eye(par.B) + gammainv*H'*(eye(par.U)-s*s'/norm(s,2)^2)*H);

% main C1PO algorithm loop
for i=2:par.C1PO.iterations
    x = par.C1PO.rho*(Ainv*x);
    x = min(max(real(x),-1),1) + 1i*min(max(imag(x),-1),1);
end
x = (sign(real(x))+1i*sign(imag(x)))/sqrt(2*par.B);

% scaling factor
beta = norm(s,2)^2/(s'*H*x);

end

%% C2PO: biConvex 1-bit PrecOding with simplified processing (Algorithm 2)
function [x, beta] = C2PO(par,s,H)

% initial guess
x = H'*s;

% preprocessing with approximate inverse
tau = par.C2PO.tau; % step size
Ainvapprox = eye(par.B) - tau*H'*(eye(par.U)-s*s'/norm(s,2)^2)*H ;

% main C1PO algorithm loop
for i=2:par.C2PO.iterations
    x = par.C2PO.rho*(Ainvapprox*x);
    x = min(max(real(x),-1),1) + 1i*min(max(imag(x),-1),1);
end
x = (sign(real(x))+1i*sign(imag(x)))/sqrt(2*par.B);

% scaling factor
beta = norm(s,2)^2/(s'*H*x);

end


%% Squared inifinity-norm relaxation with Douglas-Rachford splitting
%  (SQUID) (1-bit beamforming algorithm)

function [x,beta] = SQUID(par,s,H,N0)

% -- real-valued decomposition
HR = [ real(H) -imag(H) ; imag(H) real(H) ];
sR = [ real(s) ; imag(s) ];

% -- initialization
x = zeros(par.B*2,1);
y = zeros(par.B*2,1);
gain = 1; % ADMM algorithm parameter
epsilon = 1e-5; % ADMM algorithm parameter
Ainv = inv(HR'*HR + 0.5/gain*eye(par.B*2));
sREG = Ainv*(HR'*sR);

% -- SQUID loop
for t=1:par.SQUID.iterations
    u = sREG + 0.5/gain*(Ainv*(2*x-y));
    xold = x;
    x = prox_infinityNorm2(y+u-x,2*2*par.U*par.B*N0);
    if norm(x-xold)/norm(x)<epsilon
        break;
    end
    y = y + u - x;
end


% -- extract binary solution
xRest = sign(x);
x = 1/sqrt(2*par.B)*(xRest(1:par.B,1)+1i*xRest(par.B+1:2*par.B,1));

% -- compute output gains
beta = real(x'*H'*s)/(norm(H*x,2)^2+par.U*N0);

if beta < 0
    warning('SQUID: negative precoding factor!');
end

end

%% Infinity^2 proximal operator
function [ xk ] = prox_infinityNorm2(w,lambda)

N = length(w);
wabs = abs(w);
ws = (cumsum(sort(wabs,'descend')))./(lambda+(1:N)');
alphaopt = max(ws);
if alphaopt>0
    % -- truncation step
    xk = min(wabs,alphaopt).*sign(w);
else
    xk = zeros(size(w));
end

end