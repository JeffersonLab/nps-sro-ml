#include "VTP.hh"

VTP::VTP(int nChannels, int ntime, double deltaT) : mNChannels(nChannels), mNTime(ntime), mDeltaT(deltaT) {
	resetConfig();
}

VTP::VTP(int nChannels, int ntime, double deltaT, const std::string &configFile) :
	mNChannels(nChannels),
	mNTime(ntime),
	mDeltaT(deltaT) {
	resetConfig();
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

	auto trim = [](std::string &s) {
		s.erase(0, s.find_first_not_of(" \t\r\n"));
		s.erase(s.find_last_not_of(" \t\r\n") + 1);
	};

	while (std::getline(ss, column, ',')) {
		trim(column);
		columns.push_back(column);
	}

	std::vector<std::string> required_fields = {
		"channel",
		"VTP_FIRMWARETYPE",
		"VTP_FIRMWAREVERSION",
		"VTP_W_OFFSET",
		"VTP_W_WIDTH",
		"VTP_NPS_ECALCLUSTER_HIT_DT",
		"VTP_NPS_ECALCLUSTER_SEED_THR",
		"VTP_NPS_ECALCLUSTER_NHIT_MIN",
		"VTP_NPS_ECALCLUSTER_CLUSTER_READOUT_THR",
		"VTP_NPS_ECALCLUSTER_CLUSTER_TRIGGER_THR",
		"VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_THR",
		"VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_WIDTH",
		"VTP_NPS_ECALCLUSTER_FADCMASK_MODE"
	};

	for (const auto &field : required_fields) {
		if (std::find(columns.begin(), columns.end(), field) == columns.end()) {
			std::cerr << "Error: Required field " << field << " not found in VTP config file.\n";
			return false;
		}
	}

	std::unordered_map<std::string, size_t> col_idx;
	for (size_t i = 0; i < columns.size(); ++i) {
		col_idx[columns[i]] = i;
	}

	while (std::getline(configFile, line)) {

		std::vector<std::string> values;
		std::stringstream ss(line);
		std::string value;

		while (std::getline(ss, value, ',')) {
			values.push_back(value);
		}

		if (values.size() != columns.size()) {
			std::cerr << "Column count mismatch\n";
			continue;
		}

		int ch = std::stoi(values[col_idx["channel"]]);

		if (ch < 0 || ch >= mNChannels) {
			std::cerr << "Warning: Channel number " << ch << " out of range in VTP configuration file." << std::endl;
			continue;
		}

		mConfig.firmware_type[ch] = std::stoi(values[col_idx["VTP_FIRMWARETYPE"]]);
		mConfig.firmware_ver[ch] = std::stoi(values[col_idx["VTP_FIRMWAREVERSION"]]);
		mConfig.offset[ch] = std::stoi(values[col_idx["VTP_W_OFFSET"]]);
		mConfig.width[ch] = std::stoi(values[col_idx["VTP_W_WIDTH"]]);
		mConfig.nps.cluster_hit_dt[ch] = std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_HIT_DT"]]) * mDeltaT;
		mConfig.nps.cluster_seed_thr[ch] = std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_SEED_THR"]]);
		mConfig.nps.cluster_nhits_min[ch] = std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_NHIT_MIN"]]);
		mConfig.nps.cluster_readout_thr[ch] = std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_CLUSTER_READOUT_THR"]]);
		mConfig.nps.cluster_trigger_thr[ch] = std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_CLUSTER_TRIGGER_THR"]]);
		mConfig.nps.cluster_pair_trigger_thr[ch] =
			std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_THR"]]);
		mConfig.nps.cluster_pair_trigger_width[ch] =
			std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_CLUSTER_PAIR_TRIGGER_WIDTH"]]);
		mConfig.nps.fadcmask_mode[ch] = std::stoi(values[col_idx["VTP_NPS_ECALCLUSTER_FADCMASK_MODE"]]);
	}

	configFile.close();

	return true;
}

void VTP::resetConfig() {

	mConfig.firmware_type.clear();
	mConfig.firmware_ver.clear();
	mConfig.offset.clear();
	mConfig.width.clear();
	mConfig.nps.cluster_hit_dt.clear();
	mConfig.nps.cluster_seed_thr.clear();
	mConfig.nps.cluster_nhits_min.clear();
	mConfig.nps.cluster_readout_thr.clear();
	mConfig.nps.cluster_trigger_thr.clear();
	mConfig.nps.cluster_pair_trigger_thr.clear();
	mConfig.nps.cluster_pair_trigger_width.clear();
	mConfig.nps.fadcmask_mode.clear();

	mConfig.firmware_type.resize(mNChannels, 0);
	mConfig.firmware_ver.resize(mNChannels, 0);
	mConfig.offset.resize(mNChannels, mDefaultOffset);
	mConfig.width.resize(mNChannels, mDefaultWidth);
	mConfig.nps.cluster_hit_dt.resize(mNChannels, mDefaultHitTimingWindow);
	mConfig.nps.cluster_seed_thr.resize(mNChannels, mDefaultSeedThreshold);
	mConfig.nps.cluster_nhits_min.resize(mNChannels, mDefaultMinHits);
	mConfig.nps.cluster_readout_thr.resize(mNChannels, mDefaultReadoutThreshold);
	mConfig.nps.cluster_trigger_thr.resize(mNChannels, mDefaultClusterThreshold);
	mConfig.nps.cluster_pair_trigger_thr.resize(mNChannels, mDefaultPairClusterThreshold);
	mConfig.nps.cluster_pair_trigger_width.resize(mNChannels, mDefaultPairClusterWidth);
	mConfig.nps.fadcmask_mode.resize(mNChannels, mDefaultReadoutMode);
}

void VTP::process(
	int seedChannel, int seedTime, double seedE, const std::vector<int> &gridChannels,
	const std::vector<int> &gridTimes, const std::vector<double> &gridEnergies
) {

	int nBlocks = gridEnergies.size();
	if (gridTimes.size() != nBlocks) {
		throw std::runtime_error("Error: gridEnergies and gridTimes size mismatch.");
	}
	if (gridChannels.size() != nBlocks) {
		throw std::runtime_error("Error: gridEnergies and gridChannels size mismatch.");
	}

	// filter out those blocks outside the time window.
	auto dt = mConfig.nps.cluster_hit_dt[seedChannel] / mDeltaT; // in number of bins

	std::vector<int> validChannels;
	std::vector<int> validTimes;
	std::vector<double> validEnergies;

	validChannels.push_back(seedChannel);
	validTimes.push_back(seedTime);
	validEnergies.push_back(seedE);

	for (int iBlock = 0; iBlock < nBlocks; iBlock++) {
		auto t = gridTimes[iBlock];
		if (std::abs(t - seedTime) <= dt) {
			validChannels.push_back(gridChannels[iBlock]);
			validTimes.push_back(gridTimes[iBlock]);
			validEnergies.push_back(gridEnergies[iBlock]);
		}
	}

	auto seed_thr = mConfig.nps.cluster_seed_thr[seedChannel];
	auto min_hits = mConfig.nps.cluster_nhits_min[seedChannel];
	auto cluster_thr = mConfig.nps.cluster_trigger_thr[seedChannel];
	auto pair_cluster_thr = mConfig.nps.cluster_pair_trigger_thr[seedChannel];

	// early exit if seed energy is below threshold
	if (seedE <= seed_thr) {
		return;
	}

	// local maximum requirement
	for (int i = 0; i < validEnergies.size(); i++) {
		if (validEnergies[i] > seedE) {
			return;
		}
	}

	int nHits = validEnergies.size();
	auto totalE = std::accumulate(validEnergies.begin(), validEnergies.end(), 0.0);

	mEvent.nseeds += 1;
	mEvent.clus_sizes.push_back(nHits);
	mEvent.channels.push_back(validChannels);
	mEvent.times.push_back(validTimes);
	mEvent.energies.push_back(validEnergies);

	bool tr0 = nHits >= min_hits && totalE >= cluster_thr;
	bool tr1 = false;
	bool tr2 = false;
	bool tr3 = nHits >= min_hits && totalE >= pair_cluster_thr;
	bool tr4 = false; // if there is another cluster found in the same crate (not implemented)
	bool tr5 = false;

	mEvent.trigger0.push_back(tr0);
	mEvent.trigger1.push_back(tr1);
	mEvent.trigger2.push_back(tr2);
	mEvent.trigger3.push_back(tr3);
	mEvent.trigger4.push_back(tr4);
	mEvent.trigger5.push_back(tr5);

	return;
}

void VTP::printConfig() const {
	for (int i = 0; i < mNChannels; i++) {
		std::cout << "Channel " << i << " : "
				  << " Offset = " << mConfig.offset[i] << ", Width = " << mConfig.width[i]
				  << ", Hit DT = " << mConfig.nps.cluster_hit_dt[i]
				  << ", Seed Thr = " << mConfig.nps.cluster_seed_thr[i]
				  << ", Min Hits = " << mConfig.nps.cluster_nhits_min[i]
				  << ", Readout Thr = " << mConfig.nps.cluster_readout_thr[i]
				  << ", Cluster Thr = " << mConfig.nps.cluster_trigger_thr[i]
				  << ", Pair Cluster Thr = " << mConfig.nps.cluster_pair_trigger_thr[i]
				  << ", Pair Cluster Width = " << mConfig.nps.cluster_pair_trigger_width[i]
				  << ", FADC Mask Mode = " << mConfig.nps.fadcmask_mode[i] << std::endl;
	}
}

void vtp_reco_evt::clear() {
	this->nseeds = 0;
	this->clus_sizes.clear();
	this->channels.clear();
	this->times.clear();
	this->energies.clear();
	this->trigger0.clear();
	this->trigger1.clear();
	this->trigger2.clear();
	this->trigger3.clear();
	this->trigger4.clear();
	this->trigger5.clear();
}
void VTP::resetEvent() { mEvent.clear(); }
