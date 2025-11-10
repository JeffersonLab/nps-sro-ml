#ifndef FADC250_HH
#define FADC250_HH

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

class fADC250 {
public:
	fADC250(int channels = 1080, const std::string &configFile = "");
	~fADC250();

	bool loadConfig(const std::string &filename);
	void resetConfig();

	void readWaveform(const std::vector<double> &waveform, int channel, bool usePedestal = false);
	std::vector<int> findPulses(const std::vector<double> &waveform, int channel);
	double integrateCharge(const std::vector<double> &waveform, int channel, int pulseIndex);

	void printConfig() const;
	void reset();

	const std::vector<std::vector<int>> &getPulseTimes() const { return mPulseTimes; }
	const std::vector<std::vector<double>> &getPulseCharges() const { return mPulseCharges; }

private:
	int mChannels;							// Number of channels
	double mTimePerSample;					// ns per timestamp
	std::vector<double> mChannelThresholds; // FADC250_TET per channel
	std::vector<double> mChannelGains;		// FADC250_GAIN per channel
	std::vector<double> mChannelPedestals;	// FADC250_ALLCH_PED per channel
	std::vector<int> mChannelNSAs;			// FADC250_NSA per channel
	std::vector<int> mChannelNSBs;			// FADC250_NSB per channel

	// these are additional criteria for computing integrated charge
	int mClockCycles; // No threshold crossing occurred in the past n clock cycles

	std::vector<std::vector<int>> mPulseTimes;
	std::vector<std::vector<double>> mPulseCharges;
};

#endif