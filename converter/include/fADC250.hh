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

	void readWaveform(
		const std::vector<double> &waveform, int channel, double ped_per_samp = 0.0, bool usePedestal = false,
		bool debounce = false
	);
	std::vector<int> findPulses(const std::vector<double> &waveform_adc, int channel, bool debounce = false) const;
	std::vector<int> findPulsesNaive(const std::vector<double> &waveform_adc, int channel) const;
	std::vector<int> findPulsesDebounce(const std::vector<double> &waveform_adc, int channel) const;

	double integrateCharge(const std::vector<double> &waveform_adc, int channel, int pulseIndex) const;
	void printConfig() const;
	void reset();

	const std::vector<std::vector<int>> &getPulseTimes() const { return mPulseTimes; }
	const std::vector<std::vector<double>> &getPulseEnergies() const { return mPulseEnergies; }

	constexpr static double GetAdcTomV();
	constexpr static double GetAdcTopC();
	constexpr static double GetAdcTons();

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
	std::vector<std::vector<double>> mPulseEnergies;

protected:
	// Below are fixed values used in THcRawAdcHit class in hcana
	int mMaxPulse; // number of maximum pulses to search for in a waveform, by default 4
	int mNSAT; // number of samples above threshold after crossing so that pulse is considered to be valid, by default 2
	int mPedestalSamples; // number of samples to use for pedestal calculation (start from 0-th sample), by default 4
	constexpr static int mAdcChan = 4096;  // number of ADC channels
	constexpr static int mAdcRange = 1000; //  dynamic range of the fADC in mV

	constexpr static double mAdcImpedence = 50.0;	 // FADC input impedence in units of Ohms
	constexpr static double mAdcTimeSample = 4000.0; // Length of FADC time sample in units of ps
	constexpr static double mAdcTimeRes = 0.0625;	 // FADC time resolution in units of ns
};

#endif