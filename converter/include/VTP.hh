#ifndef VTP_HH
#define VTP_HH

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

class VTP {
public:
	VTP(int nTime, double deltaT, const std::string &configFile = "");
	~VTP();

	bool loadConfig(const std::string &filename);
	void resetConfig();
	void resetArrays();

	void
	process(const std::vector<std::vector<double>> &triggerEnergies, const std::vector<std::vector<int>> &triggerTimes);

	std::vector<bool> getTriggerType(const std::vector<double> &energies, const std::vector<bool> &hits);
	void printConfig() const;
	bool isTriggered() const;

	const std::vector<bool> &getTriggeredBase() const { return mTriggeredBase; }
	const std::vector<bool> &getTrigger0() const { return mTrigger0; }
	const std::vector<bool> &getTrigger1() const { return mTrigger1; }
	const std::vector<bool> &getTrigger2() const { return mTrigger2; }
	const std::vector<bool> &getTrigger3() const { return mTrigger3; }
	const std::vector<bool> &getTrigger4() const { return mTrigger4; }
	const std::vector<bool> &getTrigger5() const { return mTrigger5; }
	const std::vector<bool> &getTriggered() const { return mTriggered; }
	const std::vector<double> &getTriggerTimes() const { return mTriggerTimes; }

private:
	int mNTime;
	double mDeltaT;
	int mTimeWindowBins;

	double mSeedThreshold;	 // VTP_NPS_ECALCLUSTER_SEED_THR;
	double mHitTimingWindow; // VTP_NPS_ECALCLUSTER_HIT_DT;
	int mMinHits;			 // VTP_NPS_ECALCLUSTER_NHIT_MIN;

	double mClusterThreshold;	  // VTP_NPS_ECALCLUSTER_CLUSTER_TRIGGER_THR;
	double mPairClusterThreshold; // VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_THR;
	double mPairClusterWidth;	  // VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_WIDTH;

	// for Sparisfication
	int mReadoutMode;		  // VTP_NPS_ECALCLUSTER_FADCMASK_MODE;
	double mReadoutThreshold; // VTP_NPS_ECALCLUSTER_CLUSTER_READOUT_THR;

	// for storing result in the same event
	std::vector<bool> mTriggeredBase;  // [nTimeWindowBins]
	std::vector<bool> mTrigger0;	   // [nTimeWindowBins]
	std::vector<bool> mTrigger1;	   // [nTimeWindowBins]
	std::vector<bool> mTrigger2;	   // [nTimeWindowBins]
	std::vector<bool> mTrigger3;	   // [nTimeWindowBins]
	std::vector<bool> mTrigger4;	   // [nTimeWindowBins]
	std::vector<bool> mTrigger5;	   // [nTimeWindowBins]
	std::vector<bool> mTriggered;	   // [nTimeWindowBins]
	std::vector<double> mTriggerTimes; // [nTimeWindowBins]

	void calcTimeWindowBins(double dt);
};

#endif // VTP_HH