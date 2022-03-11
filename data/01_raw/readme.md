Data Set: FD001
Train trajectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trajectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trajectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trajectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)



Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:

1.	T2 Total temperature at fan inlet °R
1.	T24 Total temperature at LPC outlet °R
1.	T30 Total temperature at HPC outlet °R
1.	T50 Total temperature at LPT outlet °R
1.	P2 Pressure at fan inlet psia
1.	P15 Total pressure in bypass-duct psia
1.	P30 Total pressure at HPC outlet psia
1.	Nf Physical fan speed rpm
1.	Nc Physical core speed rpm
1.	epr Engine pressure ratio (P50/P2) (unitless)
1.	Ps30 Static pressure at HPC
1.	outlet psia phi Ratio of fuel flow to Ps30 pps/psi
1.	NRf Corrected fan speed rpm
1.	NRc Corrected core speed rpm
1.	BPR Bypass Ratio (unitless)
1.	farB Burner fuel-air ratio (unitless)
1.	htBleed Bleed Enthalpy (unitless)
1.	Nf_dmd Demanded fan speed rpm
1.	PCNfR_dmd Demanded corrected fan speed rpm
1.	W31 HPT coolant bleed lbm/s
1.	W32 LPT coolant bleed lbm/s


Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
