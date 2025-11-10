#include "fADC250.hh"

fADC250::fADC250(int channels, const std::string &configFile) : mChannels(channels), mTimePerSample(4.0) {
	resetConfig();
	if (!configFile.empty()) {
		loadConfig(configFile);
	}
	mPulseTimes.resize(mChannels);
	mPulseCharges.resize(mChannels);
}

fADC250::~fADC250() {
	// Destructor
}

void fADC250::resetConfig() {
	// default configuration according to "https://hallcweb.jlab.org/wiki/images/b/b3/NPS_VTP_DAQ.pdf"
	mChannelThresholds.resize(mChannels, 10.0);
	mChannelGains.resize(mChannels, 1.0);
	mChannelPedestals.resize(mChannels, 0.0);
	mChannelNSAs.resize(mChannels, 4); // n samples before the pulse crossing threshold
	mChannelNSBs.resize(mChannels, 9); // n samples after the pulse crossing threshold

	// ensure no more than 1 pulse every 32ns can be reported (this is a bandwidth limit on the communication link from
	// FADC -> VTP) see the VTP manual section 9.2.1 for details
	mClockCycles = 7;
}

bool fADC250::loadConfig(const std::string &filename) {

	std::ifstream configFile(filename.c_str());

	if (!configFile.is_open()) {
		std::cerr << "Error opening configuration file: " << filename << std::endl;
		return false;
	}
	configFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	std::string line;

	std::vector<bool> channel_set(mChannels, false);

	while (std::getline(configFile, line)) {
		std::stringstream ss(line);
		std::string value;
		std::vector<std::string> row_data;
		while (std::getline(ss, value, ',')) {
			row_data.push_back(value);
		}

		// channel,FADC250_MODE,FADC250_COMPRESSION,FADC250_VXSREADOUT,FADC250_W_OFFSET,FADC250_W_WIDTH,FADC250_NSA,FADC250_NSB,FADC250_NPEAK,FADC250_TRG_MASK,FADC250_TRG_WIDTH,FADC250_TRG_MINTOT,FADC250_TRG_MINMULT,FADC250_ADC_MASK,FADC250_TET_IGNORE_MASK,FADC250_INVERT_MASK,FADC250_PLAYBACK_DISABLE_MASK,FADC250_TRG_MODE_MASK,FADC250_ALLCH_DAC,FADC250_ALLCH_PED,FADC250_ALLCH_TET,FADC250_ALLCH_DELAY,FADC250_ALLCH_GAIN,FADC250_SPARSIFICATION,FADC250_ACCUMULATOR_SCALER_MODE_MASK

		int channel = std::stoi(row_data[0]);
		auto FADC250_ALLCH_TET = std::stod(row_data[20]);			// MeV
		auto FADC250_ALLCH_GAIN = std::stod(row_data[22]);			// MeV
		auto FADC250_NSA = std::stod(row_data[6]) / mTimePerSample; // timestamp unit
		auto FADC250_NSB = std::stod(row_data[7]) / mTimePerSample; // timestamp unit
		auto FADC250_ALLCH_PED = std::stod(row_data[19]);			//

		if (channel < 0 || channel >= mChannels) {
			std::cerr << "Error: Channel number " << channel << " out of range in configuration file." << std::endl;
			continue;
		}

		if (std::floor(FADC250_NSA) != FADC250_NSA || std::floor(FADC250_NSB) != FADC250_NSB) {
			throw std::runtime_error(
				"Error: FADC250_NSA and FADC250_NSB must be multiples of " + std::to_string(mTimePerSample) + " ns."
			);
		}

		channel_set[channel] = true;
		mChannelThresholds[channel] = FADC250_ALLCH_TET;
		mChannelGains[channel] = FADC250_ALLCH_GAIN;
		mChannelNSAs[channel] = static_cast<int>(FADC250_NSA);
		mChannelNSBs[channel] = static_cast<int>(FADC250_NSB);
		mChannelPedestals[channel] = FADC250_ALLCH_PED;
	}

	if (std::accumulate(channel_set.begin(), channel_set.end(), 0) != mChannels) {
		std::cerr
			<< "Warning: Not all channels were set in the configuration file. Using default values for unset channels."
			<< std::endl;
	}
	configFile.close();
	return true;
}

void fADC250::readWaveform(const std::vector<double> &waveform, int channel, bool usePedestal) {
	// subtract pedestal if needed
	std::vector<double> waveform_ = waveform;
	if (usePedestal) {
		for (auto &sample : waveform_) {
			sample -= mChannelPedestals[channel];
		}
	}

	auto pulses = findPulses(waveform_, channel);
	for (const auto &p : pulses) {
		auto charge = integrateCharge(waveform_, channel, p);
		mPulseTimes[channel].push_back(p);
		mPulseCharges[channel].push_back(charge);
	}
	return;
}

std::vector<int> fADC250::findPulses(const std::vector<double> &waveform, int channel) {

	// Find the pulse crossing threshold
	auto threshold = mChannelThresholds[channel];

	// cache all samples that cross threshold
	std::vector<bool> crossed(waveform.size(), false);
	for (size_t i = 0; i < waveform.size(); i++) {
		if (waveform[i] >= threshold) {
			crossed[i] = true;
		}
	}

	std::vector<int> thresCrossIndices; // valid indices where threshold crossing occurred

	int lastPulseIndex = -mClockCycles;
	for (int i = 0; i < (int)waveform.size(); ++i) {
		if (crossed[i] && i - lastPulseIndex >= mClockCycles) {
			thresCrossIndices.push_back(i);
			lastPulseIndex = i;
		}
	}

	return thresCrossIndices;
}

double fADC250::integrateCharge(const std::vector<double> &waveform, int channel, int pulseIndex) {

	auto nsa = mChannelNSAs[channel];
	auto nsb = mChannelNSBs[channel];
	auto gain = mChannelGains[channel];

	int startIndex = pulseIndex - nsa;
	int endIndex = pulseIndex + nsb;

	// Ensure indices are within waveform bounds
	if (startIndex < 0) {
		startIndex = 0;
	}
	if (endIndex >= static_cast<int>(waveform.size())) {
		endIndex = waveform.size() - 1;
	}

	auto charge = std::accumulate(waveform.begin() + startIndex, waveform.begin() + endIndex + 1, 0.0);
	return charge * gain;
}

void fADC250::printConfig() const {
	std::cout << "FADC250 Configuration:" << std::endl;
	for (int channel = 0; channel < mChannels; ++channel) {
		std::cout << "Channel " << channel << ": "
				  << "Threshold = " << mChannelThresholds[channel] << ", "
				  << "Gain = " << mChannelGains[channel] << ", "
				  << "Pedestal = " << mChannelPedestals[channel] << ", "
				  << "NSA = " << mChannelNSAs[channel] << ", "
				  << "NSB = " << mChannelNSBs[channel] << std::endl;
	}
	return;
}

void fADC250::reset() {
	mPulseTimes.clear();
	mPulseCharges.clear();
	mPulseTimes.resize(mChannels);
	mPulseCharges.resize(mChannels);
}