#include "VTP.hh"

VTP::VTP(int nTime, double deltaT, const std::string &configFile) : mNTime(nTime), mDeltaT(deltaT), mTimeWindowBins(1) {
	resetConfig();
	resetArrays();
	if (!configFile.empty()) {
		loadConfig(configFile);
	}
}

VTP::~VTP() {}

bool VTP::loadConfig(const std::string &filename) {

	std::ifstream configFile(filename);
	if (!configFile.is_open()) {
		std::cerr << "Error opening configuration file: " << filename << std::endl;
		return false;
	}

	std::string line;
	std::getline(configFile, line); // header
	std::stringstream ss(line);

	std::vector<std::string> columns;
	std::string column;
	while (std::getline(ss, column, ',')) {
		columns.push_back(column);
	}

	while (std::getline(configFile, line)) {
		std::stringstream ss(line);
		std::string value;
		size_t i = 0;

		while (std::getline(ss, value, ',') && i < columns.size()) {
			const std::string &col = columns[i];

			if (col == "VTP_NPS_ECALCLUSTER_SEED_THR")
				mSeedThreshold = std::stod(value);
			else if (col == "VTP_NPS_ECALCLUSTER_HIT_DT")
				mHitTimingWindow = std::stod(value) * mDeltaT;
			else if (col == "VTP_NPS_ECALCLUSTER_NHIT_MIN")
				mMinHits = std::stoi(value);
			else if (col == "VTP_NPS_ECALCLUSTER_CLUSTER_TRIGGER_THR")
				mClusterThreshold = std::stod(value);
			else if (col == "VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_THR")
				mPairClusterThreshold = std::stod(value);
			else if (col == "VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_WIDTH")
				mPairClusterWidth = std::stod(value);
			else if (col == "VTP_NPS_ECALCLUSTER_FADCMASK_MODE")
				mReadoutMode = std::stoi(value);
			else if (col == "VTP_NPS_ECALCLUSTER_CLUSTER_READOUT_THR")
				mReadoutThreshold = std::stod(value);

			++i;
		}
	}

	configFile.close();

	this->calcTimeWindowBins(mHitTimingWindow);
	return true;
}

void VTP::calcTimeWindowBins(double dt) {
	auto timeWindows = static_cast<double>(mNTime) * mDeltaT / dt;
	if (std::floor(timeWindows) != timeWindows) {
		throw std::runtime_error("Error: NTime * DeltaT is not divisible by HitTimingWindow.");
	}
	mTimeWindowBins = static_cast<int>(timeWindows);
	return;
}

void VTP::resetConfig() {

	// default configuration according to "https://hallcweb.jlab.org/wiki/images/b/b3/NPS_VTP_DAQ.pdf"
	mSeedThreshold = 50.0;	 // MeV
	mHitTimingWindow = 20.0; // ns
	mMinHits = 1;
	mClusterThreshold = 900.0;	   // MeV
	mPairClusterThreshold = 500.0; // MeV
	mPairClusterWidth = 20.0;	   // ns

	// for Sparisfication
	mReadoutMode = 7;		   // 0 for 5x5 or 1 for 7x7
	mReadoutThreshold = 100.0; // MeV

	this->calcTimeWindowBins(mHitTimingWindow);
}

std::vector<bool> VTP::getTriggerType(const std::vector<double> &energies, const std::vector<bool> &hits) {

	std::vector<bool> triggers(7, false); // base trigger, trigger 0-5

	auto seedE = energies[0]; // seed block is 0-th index
	auto maxE = *std::max_element(energies.begin(), energies.end());

	// base trigger
	// Central block has energy >= VTP_NPS_ECALCLUSTER_SEED_THR and is the maximum energy block in the grid
	if (seedE >= mSeedThreshold && std::abs(seedE - maxE) < 1e-9) {
		triggers[0] = true;
	}

	// clus trigger -> readout
	auto totalE = std::accumulate(energies.begin(), energies.end(), 0.0);
	int nHits = std::count(hits.begin(), hits.end(), true);

	if (nHits >= mMinHits && totalE >= mClusterThreshold) {
		triggers[0] = true; // cluster trigger

		if (totalE >= mPairClusterThreshold) {
			triggers[3] = true; // local pair 1
								// if there is another cluster seed in the same crate
								// VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_WIDTH
								// triggers[4] = true; // local pair 2
		}
	}

	return triggers;
}

void VTP::process(
	const std::vector<std::vector<double>> &gridEnergies, const std::vector<std::vector<int>> &gridTimes
) {
	resetArrays();

	// input = blocks of energies and times within the entire event time window
	// 0-th index is the seed block

	int nBlocks = gridEnergies.size();
	if (gridTimes.size() != nBlocks) {
		throw std::runtime_error("Error: gridEnergies and gridTimes size mismatch.");
	}

	// divide the vectors into time slices
	// mTimeWindowBins = 22 (440 / 20);

	for (int iWindow = 0; iWindow < mTimeWindowBins; iWindow++) {

		int startTime = iWindow * mHitTimingWindow / mDeltaT;
		int endTime = (iWindow + 1) * mHitTimingWindow / mDeltaT;

		std::vector<bool> hitBlocks(nBlocks, false);
		std::vector<double> hitEnergies(nBlocks, 0.0); // fadc250 only allow 1 hit within the time window
		std::vector<double> hitTimes(nBlocks, 0.0);

		for (int iBlock = 0; iBlock < nBlocks; iBlock++) {

			for (int iHit = 0; iHit < gridTimes[iBlock].size(); iHit++) {
				auto t = gridTimes[iBlock][iHit];

				// auto tns = t * mDeltaT;
				// if (tns < 50 || tns > 370) {
				//     continue; // only consider hits within [50, 370] ns
				// }

				if (t >= startTime && t < endTime) {
					hitBlocks[iBlock] = true;
					if (hitEnergies[iBlock] > 0.0) {
						throw std::runtime_error("Error: multiple hits in one time window not allowed.");
					}

					hitEnergies[iBlock] += gridEnergies[iBlock][iHit];
					hitTimes[iBlock] = t;
				}
			}
		}

		auto triggerType = getTriggerType(hitEnergies, hitBlocks); // {0,0,0,0,0,0,0}, {1,0,0,1,0,0}, etc ...
		mTriggeredBase[iWindow] = triggerType[0];
		mTrigger0[iWindow] = triggerType[1];
		mTrigger1[iWindow] = triggerType[2];
		mTrigger2[iWindow] = triggerType[3];
		mTrigger3[iWindow] = triggerType[4];
		mTrigger4[iWindow] = triggerType[5];
		mTrigger5[iWindow] = triggerType[6];
		mTriggered[iWindow] = std::any_of(triggerType.begin(), triggerType.end(), [](bool v) { return v; });

		if (mTriggered[iWindow]) {
			mTriggerTimes[iWindow] = hitTimes[0]; // seed block time
		}
	}
	return;
}

void VTP::printConfig() const {
	std::cout << "VTP Configuration:" << std::endl;
	std::cout << "Seed Threshold: " << mSeedThreshold << " MeV" << std::endl;
	std::cout << "Hit Timing Window: " << mHitTimingWindow << " ns" << std::endl;
	std::cout << "Minimum Hits: " << mMinHits << std::endl;
	std::cout << "Cluster Threshold: " << mClusterThreshold << " MeV" << std::endl;
	std::cout << "Pair Cluster Threshold: " << mPairClusterThreshold << " MeV" << std::endl;
	std::cout << "Pair Cluster Width: " << mPairClusterWidth << " ns" << std::endl;
	std::cout << "Readout Mode: " << mReadoutMode << std::endl;
	std::cout << "Readout Threshold: " << mReadoutThreshold << " MeV" << std::endl;
}

void VTP::resetArrays() {
	mTrigger0.clear();
	mTrigger1.clear();
	mTrigger2.clear();
	mTrigger3.clear();
	mTrigger4.clear();
	mTrigger5.clear();
	mTriggered.clear();
	mTriggerTimes.clear();

	mTriggeredBase.resize(mTimeWindowBins, false);
	mTrigger0.resize(mTimeWindowBins, false);
	mTrigger1.resize(mTimeWindowBins, false);
	mTrigger2.resize(mTimeWindowBins, false);
	mTrigger3.resize(mTimeWindowBins, false);
	mTrigger4.resize(mTimeWindowBins, false);
	mTrigger5.resize(mTimeWindowBins, false);
	mTriggered.resize(mTimeWindowBins, false);
	mTriggerTimes.resize(mTimeWindowBins, std::numeric_limits<double>::max());

	return;
}

bool VTP::isTriggered() const {
	return std::any_of(mTriggeredBase.begin(), mTriggeredBase.end(), [](bool v) { return v; });
}