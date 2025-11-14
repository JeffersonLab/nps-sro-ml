#include "fADC250.hh"

fADC250::fADC250(int channels, const std::string &configFile) :
	mChannels(channels),
	mTimePerSample(4.0),
	mPedestalSamples(4),
	mMaxPulse(4),
	mNSAT(2) {
	resetConfig();
	if (!configFile.empty()) {
		loadConfig(configFile);
	}
	mPulseTimes.resize(mChannels);
	mPulseEnergies.resize(mChannels);
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
		auto FADC250_ALLCH_TET = std::stod(row_data[20]);			// ADC unit
		auto FADC250_ALLCH_GAIN = std::stod(row_data[22]);			// ADC * GAIN --> MeV
		auto FADC250_NSA = std::stod(row_data[6]) / mTimePerSample; // timestamp unit
		auto FADC250_NSB = std::stod(row_data[7]) / mTimePerSample; // timestamp unit
		auto FADC250_ALLCH_PED = std::stod(row_data[19]);			// mV

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

void fADC250::readWaveform(
	const std::vector<double> &waveform, int channel, double ped_per_samp, bool usePedestal, bool debounce
) {
	std::vector<double> waveform_adc = waveform;

	if (usePedestal) {
		for (auto &sample : waveform_adc) {
			sample += ped_per_samp * mPedestalSamples;
			sample -= mChannelPedestals[channel];
		}
	}

	for (auto &sample : waveform_adc) {
		sample /= GetAdcTomV();
	}

	auto pulses = findPulses(waveform_adc, channel, debounce);
	for (const auto &p : pulses) {
		auto charge = integrateCharge(waveform_adc, channel, p);
		mPulseTimes[channel].push_back(p);
		mPulseEnergies[channel].push_back(charge);
	}
	return;
}

std::vector<int> fADC250::findPulses(const std::vector<double> &waveform_adc, int channel, bool debounce) const {
	if (debounce) {
		return findPulsesDebounce(waveform_adc, channel);
	} else {
		return findPulsesNaive(waveform_adc, channel);
	}
}

std::vector<int> fADC250::findPulsesDebounce(const std::vector<double> &waveform_adc, int channel) const {

	auto thres = mChannelThresholds[channel];

	std::vector<int> res;
	bool prev_above = false; // debounce
	int npulse = 0;
	int ns = 0;

	while (ns < (int)waveform_adc.size() && npulse < mMaxPulse) {
		if (prev_above) {
			if (waveform_adc[ns] < thres) {
				prev_above = false;
			}
			ns++;
			continue;
		}

		// Check mNSAT consecutive samples above threshold
		int n_above = 0;
		for (int k = 0; k < mNSAT && (ns + k) < (int)waveform_adc.size(); k++) {
			if (waveform_adc[ns + k] >= thres) {
				n_above++;
			}
		}

		if (n_above >= mNSAT) {
			// Valid pulse found
			res.push_back(ns);
			prev_above = true;
			npulse++;
			ns += mClockCycles;
		} else {
			ns++;
		}
	}

	return res;
}

std::vector<int> fADC250::findPulsesNaive(const std::vector<double> &waveform_adc, int channel) const {

	auto threshold = mChannelThresholds[channel];

	std::vector<int> res; // valid indices where threshold crossing occurred
	int lastPulseIndex = -mClockCycles;
	for (int i = 0; i < (int)waveform_adc.size(); ++i) {

		auto is_crossed = waveform_adc[i] >= threshold;

		if (is_crossed && i - lastPulseIndex >= mClockCycles) {
			res.push_back(i);
			lastPulseIndex = i;
		}
	}
	return res;
}

double fADC250::integrateCharge(const std::vector<double> &waveform_adc, int channel, int pulseIndex) const {

	auto nsa = mChannelNSAs[channel];
	auto nsb = mChannelNSBs[channel];
	auto gain = mChannelGains[channel];

	int startIndex = pulseIndex - nsb;
	int endIndex = pulseIndex + nsa - 1;

	// Ensure indices are within waveform bounds
	if (startIndex < 0) {
		startIndex = 0;
	}
	if (endIndex >= static_cast<int>(waveform_adc.size())) {
		endIndex = waveform_adc.size() - 1;
	}

	auto integral = std::accumulate(waveform_adc.begin() + startIndex, waveform_adc.begin() + endIndex + 1, 0.0);
	// GAIN is setup such that GAIN * adc unit --> MeV.
	// NOTE from Ben : gain precision is up to 0.004, will have adjust the value from config file
	return integral * gain;
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
	mPulseEnergies.clear();
	mPulseTimes.resize(mChannels);
	mPulseEnergies.resize(mChannels);
}

constexpr double fADC250::GetAdcTomV() {
	// 1000 mV / 4096 ADC channels
	return static_cast<double>(mAdcRange) / mAdcChan;
}

// Convert integral to pC
constexpr double fADC250::GetAdcTopC() {
	// (1 V / 4096 adc channels) * (4000 ps time sample / 50 ohms input resistance) = 0.020 pc/channel
	return (static_cast<double>(mAdcRange) / mAdcChan) * (mAdcTimeSample / mAdcImpedence);
}

// Convert time sub samples to ns
constexpr double fADC250::GetAdcTons() { return mAdcTimeRes; }