% ADCPDEMO Demo script for reading ADCP data
%
%******************Demo for ADCP reading programs*******************
%
% This package consists of two functions that read binary data generated by
% RDI ADCPs into a matlab data structure. Some averaging and subsampling
% can be performed but after that data analysis and visualization is
% up to you.
%
% Two kinds of RDI files can be read with this software:
%
%   1) "Raw" files (usually downloaded from moored instruments) using
%      the BBDF format (compatible with BB and WH ADCPs) for (hopefully
%      all) firmware versions <=16.19.
%
%      Note that slight modifications of this "raw" format are used
%      by WINRIVER and VMDAS.
%      
%   2) "P-files" (processed files generated by TRANSECT or WINRIVER after
%      processing data acquired in real-time. P-files are preferable 
%      in this case because navigation data and configuration data is
%      merged with the raw data). NB, BB, and WH instrument files can
%      be read.
%
%
% This demo script will provide some examples of how to use the code.
%
% WARNING: RDI is constantly improving their products, and so the format
%          of their binary files is constantly changing. I have done the
%          best I can to track down all firmware and program version
%          dependencies and this code has been tested on datasets from
%          a large-ish number of researchers and instruments but it
%          may still fail on your file. If it does, please let me know
%          so I can figure out why!
%
% Author: R. Pawlowicz (rich@eos.ubc.ca, www.eos.ubc.ca/~rich/research.html)
%         [some support for parts of this work was provided by RESCAN
%          ENVIRONMENTAL SERVICES www.rescan.com]

% Changes - clc instead of escape code wizardry for clearing screen , thanks to F. Bahr
 more on
 help adcpdemo

fprintf('--> Now we being the demonstration.....\n');
disp('press any key to continue');  
pause;   
clc;

fprintf('--> Default read (5-ping averages of whole file) \n');
fprintf('--> [This data segment was provided by R. Dewey] \n');
fprintf('\n\necho on\n');
echo on;
adcp=rdradcp('mooredwh-bbdf.000');
echo off;


adcp
fprintf('--> Once read this is what the ADCP data structure looks like!\n');
disp('press any key to continue');  
pause;   
clc;


adcp.config
fprintf('--> Configuration data is stored in a sub-structure "adcp.config"\n');
fprintf('--> Note that this particular dataset was recorded in\n');
fprintf('--> Beam coordinates by someone who preferred doing the\n');
fprintf('--> velocity mapping themselves for this deployment\n');
disp('press any key to continue');  
pause; 
clc;  
fprintf('echo on\n');
echo on;
clf
subplot(311);
plot(adcp.number,adcp.roll);
ylabel('Roll angle');
title('Deployment of ADCP');

subplot(312);
plot(adcp.number,adcp.pitch);
ylabel('Pitch angle');

subplot(313);
pcolor(adcp.number,adcp.config.ranges,squeeze(mean(adcp.intens,2)));
shading flat;
ylabel('Range (m)');
shg;

echo off;
fprintf('--> A simple check of some of the raw data to see how\n');
fprintf('--> the instrument behaved on deployment \n');
fprintf('--> (the surface can be seen in mean backscatter) \n\n\n');

fprintf('--> Now let''s load some real-time data - we shall read\n');
fprintf('--> the first 100 pings of the file, NOT use bottom or\n'); 
fprintf('--> navigation as a reference, and perform a de-spiked 10-ping average. \n');

disp('press any key to continue');  
pause;   
fprintf('\n\necho on\n');

echo on;
adcp=rdpadcp('realtimewh-p.000',10,100,'ref','none','despike','yes');

clf
plot(adcp.north_vel,adcp.config.range);
set(gca,'ydir','reverse');
xlabel('Velocity (m/s)');ylabel('depth (m)');
title('ADCP NORTH VELOCITY','fontsize',16);
shg;

echo off
fprintf('--> ...And we plot the north velocity of all pings.\n');
fprintf('--> ..Simple!\n'); 
fprintf('-->               - Enjoy, RP. \n');
	     	     
more off
  
    
        
