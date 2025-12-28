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

typedef struct {
	std::vector<double> thr;  // FADC250_TET per channel
	std::vector<double> gain; // FADC250_GAIN per channel
	std::vector<double> ped;  // FADC250_ALLCH_PED per channel
	std::vector<int> nsa;	  // FADC250_NSA per channel
	std::vector<int> nsb;	  // FADC250_NSB per channel
} fadc250_cfg;

typedef struct {
	int nhits;
	std::vector<int> times;
	std::vector<double> energies;
	std::vector<int> channels;
} fadc250_evt;

class fADC250 {
public:
	fADC250(int channels = 1080, const std::string &configFile = "");
	~fADC250();

	bool loadConfig(const std::string &filename);
	void resetConfig();
	void resetEvent();
	void printConfig() const;
	void processRawWaveform(const std::vector<double> &waveform, int channel, bool debounce = false);

	const fadc250_evt &getEvent() const { return mEvent; }

	constexpr static double GetAdcTomV();
	constexpr static double GetAdcTopC();
	constexpr static double GetAdcTons();

private:
	int mChannels;		   // Number of channels
	double mTimePerSample; // ns per timestamp

	fadc250_cfg mConfig;
	fadc250_evt mEvent;

	int mClockCycles; // No threshold crossing occurred in the past n clock cycles

	std::vector<int> findPulses(const std::vector<double> &waveform_adc, double thr, int opt) const;
	std::vector<int> findPulseBR(const std::vector<double> &waveform_adc, double thr) const;
	std::vector<int> findPulsesNaive(const std::vector<double> &waveform_adc, double thr) const;
	std::vector<int> findPulsesDebounce(const std::vector<double> &waveform_adc, double thr) const;
	double integrateCharge(const std::vector<double> &waveform_adc, int channel, int pulseIndex) const;

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

	// default configuration according to "https://hallcweb.jlab.org/wiki/images/b/b3/NPS_VTP_DAQ.pdf"
	double mDefaultChannelThresholds = 10.0;
	double mDefaultChannelGains = 1.0;
	double mDefaultChannelPedestals = 0.0;
	int mDefaultChannelNSAs = 4; // n samples before the pulse crossing threshold
	int mDefaultChannelNSBs = 9; // n samples after the pulse crossing threshold

	// ensure no more than 1 pulse every 32ns can be reported (this is a bandwidth limit on the communication link from
	// FADC -> VTP) see the VTP manual section 9.2.1 for details
	int mDefaultClockCycles = 7;
};

#endif