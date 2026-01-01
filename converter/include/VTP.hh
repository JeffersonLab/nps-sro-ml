#ifndef VTP_HH
#define VTP_HH

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

typedef struct {
	std::vector<int> firmware_type;
	std::vector<int> firmware_ver;
	std::vector<int> offset;
	std::vector<int> width;

	struct {
		// Clustering Config
		std::vector<int> cluster_hit_dt;
		std::vector<int> cluster_seed_thr;
		std::vector<int> cluster_nhits_min;
		std::vector<int> cluster_readout_thr;
		std::vector<int> cluster_trigger_thr;
		std::vector<int> cluster_pair_trigger_thr;
		std::vector<int> cluster_pair_trigger_width;
		std::vector<int> fadcmask_mode;
	} nps;

} vtp_cfg;

typedef struct {
	int nseeds;
	std::vector<int> clus_sizes;			   // [nseeds]
	std::vector<std::vector<int>> channels;	   // [nseeds][blocks_in_cluster]
	std::vector<std::vector<int>> times;	   // [nseeds][blocks_in_cluster]
	std::vector<std::vector<double>> energies; // [nseeds][blocks_in_cluster]

	std::vector<bool> trigger0; // [nseeds]
	std::vector<bool> trigger1; // [nseeds]
	std::vector<bool> trigger2; // [nseeds]
	std::vector<bool> trigger3; // [nseeds]
	std::vector<bool> trigger4; // [nseeds]
	std::vector<bool> trigger5; // [nseeds]

	void clear();

} vtp_reco_evt;

class VTP {
public:
	VTP(int nChannels, int ntime, double deltaT);
	VTP(int nChannels, int ntime, double deltaT, const std::string &configFile);
	~VTP();

	bool loadConfig(const std::string &filename);
	void printConfig() const;
	void resetConfig();
	void resetEvent();

	void process(
		int seedChannel, int seedTime, double seedE, const std::vector<int> &gridChannels,
		const std::vector<int> &gridTimes, const std::vector<double> &gridEnergies
	);

	const vtp_reco_evt &getEvent() const { return mEvent; }

private:
	int mNChannels;
	int mNTime;
	double mDeltaT;

	vtp_cfg mConfig;
	vtp_reco_evt mEvent;

protected:
	// default configuration according to "https://hallcweb.jlab.org/wiki/images/b/b3/NPS_VTP_DAQ.pdf"

	double mDefaultFAdcOffset = 4500; // FADC Lookback time from trigger in ns
	double mDefaultFAdcWidth = 440;	  // Waveform readout window in ns
	double mDefaultOffset = 4448;	  // VTP Lookback time time from trigger in ns
	double mDefaultWidth = 1000;	  // Window width to find clusters in ns

	double mDefaultSeedThreshold = 50.0;		 // Threshold for defining a cluster seed in MeV
	double mDefaultHitTimingWindow = 20.0;		 // Coincidence window for cluster formation in ns
	int mDefaultMinHits = 1;					 // Mininum # Hits to define cluster
	double mDefaultClusterThreshold = 900.0;	 // Single-cluster validation threshold (Bit 0) MeV
	double mDefaultPairClusterThreshold = 500.0; // Two-cluster validation threshold MeV
	double mDefaultPairClusterWidth = 20.0;		 //  Output width of VTP Bit 4 ns

	// for Sparisfication
	int mDefaultReadoutMode = 7;			 // 0 for 5x5 or 1 for 7x7
	double mDefaultReadoutThreshold = 100.0; //  Cluster threshold for readout in MeV
};

#endif // VTP_HH