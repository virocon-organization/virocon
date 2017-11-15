%% Example 1: 2-dimensional contour

ModelArray(1).name = 'HDC2d_WN';
ModelArray(1).gridCenterPoints = {0:0.05:20 0:0.01:20};
ModelArray(1).distribution = 'weibull';
ModelArray(1).isConditional = [0 0 0];
ModelArray(1).coeff = {2.776 1.471 0.8888};
ModelArray(1).label = 'significant wave height [m]';
ModelArray(2).distribution = 'normal';
ModelArray(2).isConditional = [1 1];
ModelArray(2).coeff = { @(x1)4 + 10 * x1^0.02;
                    @(x1)0.1 + 0.02 * exp(-0.1*x1)};
ModelArray(2).label = 'zero-upcrossing period [s]';

% Define an exceedance probability, alpha, based on a return period, nYears
nYears = 50;
n = nYears * 365.25 * 24/3;
alpha = 1/n;

% Define the grid center points for the numerical integration (each ...
% cell represents one dimension)
gridCenterPoints = ModelArray.gridCenterPoints;

% Calculate the highest density contour and plot it
[fm, x1Hdc, x2Hdc, x3Hdc, x4Hdc] = computeHdc(ModelArray, alpha, ...
    gridCenterPoints, 1);

M = [x1Hdc{1}, x2Hdc{1}];
dlmwrite('C:\01_data\kai-lukas_windmeier\hdc2d_wn.csv', M)

%% HDC3d_WLN
clear ModelArray

ModelArray(1).name = 'HDC3d_WLN';
ModelArray(1).gridCenterPoints = {0:0.5:20 0:0.5:20 0:0.05:20};
ModelArray(1).distribution = 'weibull';
ModelArray(1).isConditional = [0 0 0];
ModelArray(1).coeff = {2.776 1.471 0.8888};
ModelArray(1).label = 'significant wave height [m]';
ModelArray(2).distribution = 'lognormal';
ModelArray(2).isConditional = [1 1];
ModelArray(2).coeff = { @(x1)0.1 + 1.5 * x1^0.2;
                    @(x1)0.1 + 0.2 * exp(-0.2*x1)};
ModelArray(2).label = 'zero-upcrossing period [s]';
ModelArray(3).distribution = 'normal';
ModelArray(3).isConditional = [1 1];
ModelArray(3).coeff = { @(x1,x2)4 + 10 * x1^0.02;
                    @(x1,x2)0.1 + 0.02 * exp(-0.1*x1)};
ModelArray(3).label = 'x3 [-]';

% Define an exceedance probability, alpha, based on a return period, nYears
nYears = 50;
n = nYears * 365.25 * 24/3;
alpha = 1/n;

% Define the grid center points for the numerical integration (each ...
% cell represents one dimension)
gridCenterPoints = ModelArray.gridCenterPoints;

% Calculate the highest density contour and plot it
[fm, x1Hdc, x2Hdc, x3Hdc, x4Hdc] = computeHdc(ModelArray, alpha, ...
    gridCenterPoints, 1);

M = [x1Hdc{1}', x2Hdc{1}', x3Hdc{1}'];
dlmwrite('C:\01_data\kai-lukas_windmeier\hdc3d_wln.csv', M)

%% Iform2d_WL

% Define a joint probability distribution (taken from Vanemn and Bitner-
% Gregersen, 2012):
ModelArray(1).distribution = 'weibull';
ModelArray(1).isConditional = [0 0 0];
ModelArray(1).coeff = {2.776 1.471 0.8888};
ModelArray(2).distribution = 'lognormal';
ModelArray(2).isConditional = [1 1];
ModelArray(2).coeff = { @(x1Array)0.1000 + 1.489 * x1Array^0.1901;
    @(x1Array)0.0400 + 0.1748 * exp(-0.2243*x1Array)};

% Either use the default parameters: 
[x1Array, x2Array] = computeIformContour(ModelArray, 4.34878707, 400, 1, 0); 

M = [x1Array', x2Array'];
dlmwrite('C:\01_data\kai-lukas_windmeier\iform2d_wl.csv', M)

%% Iform2d_WN

% Define a joint probability distribution (taken from Vanemn and Bitner-
% Gregersen, 2012):
ModelArray(1).distribution = 'weibull';
ModelArray(1).isConditional = [0 0 0];
ModelArray(1).coeff = {2.776 1.471 0.8888};
ModelArray(2).distribution = 'normal';
ModelArray(2).isConditional = [1 1];
ModelArray(2).coeff = { @(x1Array)7 + 1.489 * x1Array^0.1901;
    @(x1Array)1.5 + 0.1748 * exp(-0.2243*x1Array)};

% Either use the default parameters: 
[x1Array, x2Array] = computeIformContour(ModelArray, 4.34878707, 400, 1, 0); 

M = [x1Array', x2Array'];
dlmwrite('C:\01_data\kai-lukas_windmeier\iform2d_wn.csv', M)