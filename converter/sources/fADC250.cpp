#include "fADC250.hh"

fADC250::fADC250(int channels, const std::string &configFile) :
	mChannels(channels),
	mTimePerSample(4.0),
	mPedestalSamples(4),
	mMaxPulse(4),
	mNSAT(2) {
	resetEvent();
	resetConfig();
	if (!configFile.empty()) {
		loadConfig(configFile);
	}
}

fADC250::~fADC250() {
	// Destructor
}

void fADC250::resetConfig() {
	mConfig.thr.resize(mChannels, mDefaultChannelThresholds);
	mConfig.gain.resize(mChannels, mDefaultChannelGains);
	mConfig.ped.resize(mChannels, mDefaultChannelPedestals);
	mConfig.nsa.resize(mChannels, mDefaultChannelNSAs);
	mConfig.nsb.resize(mChannels, mDefaultChannelNSBs);
	mClockCycles = mDefaultClockCycles;
}

void fADC250::resetEvent() {
	mEvent.nhits = 0;
	mEvent.times.clear();
	mEvent.energies.clear();
	mEvent.channels.clear();
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
		auto FADC250_ALLCH_PED = std::stod(row_data[19]);			// ADC unit

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
		mConfig.thr[channel] = FADC250_ALLCH_TET;
		mConfig.gain[channel] = FADC250_ALLCH_GAIN;
		mConfig.nsa[channel] = static_cast<int>(FADC250_NSA);
		mConfig.nsb[channel] = static_cast<int>(FADC250_NSB);
		mConfig.ped[channel] = FADC250_ALLCH_PED;
	}

	if (std::accumulate(channel_set.begin(), channel_set.end(), 0) != mChannels) {
		std::cerr
			<< "Warning: Not all channels were set in the configuration file. Using default values for unset channels."
			<< std::endl;
		resetConfig();
	}
	configFile.close();
	return true;
}

void fADC250::processRawWaveform(const std::vector<double> &waveform, int channel, bool debounce) {

	auto ped = mConfig.ped[channel];
	auto thr = mConfig.thr[channel];

	std::vector<double> waveform_ = waveform;
	for (auto &sample : waveform_) {
		sample -= ped;
	}

	auto pulses = findPulses(waveform_, thr, debounce);

	mEvent.nhits += pulses.size();

	for (const auto &p : pulses) {
		auto charge = integrateCharge(waveform_, channel, p);
		mEvent.times.push_back(p);
		mEvent.energies.push_back(charge);
		mEvent.channels.push_back(channel);
	}
	return;
}

std::vector<int> fADC250::findPulses(const std::vector<double> &waveform_adc, double thr, int opt) const {
	switch (opt) {
	case 0:
		return findPulseBR(waveform_adc, thr);
	case 1:
		return findPulsesDebounce(waveform_adc, thr);
	case 2:
		return findPulsesNaive(waveform_adc, thr);
	default:
		throw std::invalid_argument("Invalid option for findPulses");
	}
}

std::vector<int> fADC250::findPulsesDebounce(const std::vector<double> &waveform_adc, double thr) const {

	std::vector<int> res;
	bool prev_above = false; // debounce
	int npulse = 0;
	int ns = 0;

	while (ns < (int)waveform_adc.size() && npulse < mMaxPulse) {
		if (prev_above) {
			if (waveform_adc[ns] < thr) {
				prev_above = false;
			}
			ns++;
			continue;
		}

		// Check mNSAT consecutive samples above threshold
		int n_above = 0;
		for (int k = 0; k < mNSAT && (ns + k) < (int)waveform_adc.size(); k++) {
			if (waveform_adc[ns + k] >= thr) {
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

std::vector<int> fADC250::findPulseBR(const std::vector<double> &waveform_adc, double thr) const {

	int current_over, last_over = 0;
	std::vector<int> res;
	for (int i = 0; i < (int)waveform_adc.size(); ++i) {

		if (waveform_adc[i] > thr) {
			current_over = 1;
		} else {
			current_over = 0;
		}

		if (current_over && !last_over) {
			res.push_back(i);
		}
		last_over = current_over;
	}
	return res;
}

std::vector<int> fADC250::findPulsesNaive(const std::vector<double> &waveform_adc, double thr) const {

	std::vector<int> res; // valid indices where threshold crossing occurred
	int lastPulseIndex = -mClockCycles;
	for (int i = 0; i < (int)waveform_adc.size(); ++i) {

		auto is_crossed = waveform_adc[i] >= thr;

		if (is_crossed && i - lastPulseIndex >= mClockCycles) {
			res.push_back(i);
			lastPulseIndex = i;
		}
	}
	return res;
}

double fADC250::integrateCharge(const std::vector<double> &waveform_adc, int channel, int pulseIndex) const {

	auto nsa = mConfig.nsa[channel];
	auto nsb = mConfig.nsb[channel];
	auto gain = mConfig.gain[channel];

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
	// return integral * gain;
	auto e = integral * gain;

	// hardware clip to 13 bits
	if (e < 0) {
		e = 0;
	}
	if (e > 8191) {
		e = 8191;
	}
	return e;
}

void fADC250::printConfig() const {
	std::cout << "FADC250 Configuration:" << std::endl;
	for (int channel = 0; channel < mChannels; ++channel) {
		std::cout << "Channel " << channel << ": "
				  << "Threshold = " << mConfig.thr[channel] << ", "
				  << "Gain = " << mConfig.gain[channel] << ", "
				  << "Pedestal = " << mConfig.ped[channel] << ", "
				  << "NSA = " << mConfig.nsa[channel] << ", "
				  << "NSB = " << mConfig.nsb[channel] << std::endl;
	}
	return;
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