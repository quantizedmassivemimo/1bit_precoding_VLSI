=======================================================
Simulator for 1-bit Massive MU-MIMO Precoding in VLSI with CxPO
-------------------------------------------------------
(c) 2017 Christoph Studer, Oscar Castañeda, and Sven Jacobsson
e-mail: studer@cornell.edu, oc66@cornell.edu, & sven.jacobsson@ericsson.com
=======================================================

# Important information:

If you are using the simulator (or parts of it) for a publication, then you MUST cite our paper:

Oscar Castañeda, Sven Jacobsson, Giuseppe Durisi, Mikael Coldrey, Tom Goldstein, and Christoph Studer, "1-bit Massive MU-MIMO Precoding in VLSI," IEEE Journal on Emerging and Selected Topics in Circuits and Systems (JETCAS), to appear in 2017

and clearly mention this in your paper. More information about our research can be found at: http://vip.ece.cornell.edu

# How to start a simulation:

Simply run  

>> precoder_sim

which starts a simulation in a 256 BS antennas, 16 users massive MIMO system with 16-QAM modulation using ZF and MRT precoding (both infinite precision and 1-bit quantization), 1-bit SQUID precoding, as well as C1PO and C2PO as detailed in the paper. 


The simulator runs with predefined parameters. You can provide your own system and simulation parameters by passing your own "par"-structure (see the simulator for an example). Note that we use default parameters for the considered system configuration; if you want to run the simulation with other parameters, please refer to the MATLAB code for other parameter settings. 

We highly recommend you to execute the code step-by-step (using MATLAB's debug mode) in order to get a detailed understanding of the simulator. 

# Version 0.1 (August 14, 2017) - studer@cornell.edu - initial release for GitHub