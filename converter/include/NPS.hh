#ifndef NPS_HH
#define NPS_HH

#include "TChain.h"
#include <array>
#include <iostream>
#include <string>
#include <unordered_set>

namespace NPS {

const int NTIME = 110;						// number of time samples for each fADC channel
const double DELTA_T = 4.0;					// time in ns per timestamp
const int NCOLS = 30;						// number of calorimeter columns
const int NROWS = 36;						// number of calorimeter blocks in each column
const int NBLOCKS = NCOLS * NROWS;			// number of calorimeter blocks
const int NSLOTS = 1104;					// number of maximum slots
const int NDATA = NSLOTS * (NTIME + 1 + 1); //(1104 slots fADC for each of 110 timestamp + 2 pmt

const int MAX_CLUSTERS = 256;	 // arbitrary limit on the number of clusters found in hcana
const int MAX_VTP_TRIGGERS = 64; // arbitrary limit on the number of VTP triggers

struct BlockInfo {
	int channel;
	int row;
	int col;
	int crate;
	int slot;
};

struct PairHash {
	size_t operator()(const std::pair<int, int> &p) const noexcept {
		return (static_cast<size_t>(p.first) << 32) ^ static_cast<int>(p.second);
	}
};

class Geometry {
public:
	Geometry(const std::string &config_file);
	~Geometry() = default;

	int getBlockFromColRow(int col, int row) const;
	std::pair<int, int> getColRowFromBlock(int block) const;
	int getCrateFromBlock(int block) const;
	int getSlotFromBlock(int block) const;

private:
	std::vector<BlockInfo> mBlocks;				  // index-based lookup
	std::unordered_map<int, BlockInfo> mIndexMap; // block -> info
	std::unordered_map<std::pair<int, int>, int, PairHash> mRcToIndex;
	void loadConfig(const std::string &config_file);
};

struct npsBranches {

	// Global branches
	double g_evtime;

	// Ndata.NPS branches
	int Ndata_NPS_cal_clusE;
	int Ndata_NPS_cal_clusSize;
	int Ndata_NPS_cal_clusT;
	int Ndata_NPS_cal_clusX;
	int Ndata_NPS_cal_clusY;
	int Ndata_NPS_cal_clusZ;
	int Ndata_NPS_cal_fly_adcCounter;
	int Ndata_NPS_cal_fly_adcErrorFlag;
	int Ndata_NPS_cal_fly_adcPed;
	int Ndata_NPS_cal_fly_adcPedRaw;
	int Ndata_NPS_cal_fly_adcPulseAmp;
	int Ndata_NPS_cal_fly_adcPulseAmpRaw;
	int Ndata_NPS_cal_fly_adcPulseInt;
	int Ndata_NPS_cal_fly_adcPulseIntRaw;
	int Ndata_NPS_cal_fly_adcPulseTime;
	int Ndata_NPS_cal_fly_adcPulseTimeRaw;
	int Ndata_NPS_cal_fly_adcSampPed;
	int Ndata_NPS_cal_fly_adcSampPedRaw;
	int Ndata_NPS_cal_fly_adcSampPulseAmp;
	int Ndata_NPS_cal_fly_adcSampPulseAmpRaw;
	int Ndata_NPS_cal_fly_adcSampPulseInt;
	int Ndata_NPS_cal_fly_adcSampPulseIntRaw;
	int Ndata_NPS_cal_fly_adcSampPulseTime;
	int Ndata_NPS_cal_fly_adcSampPulseTimeRaw;
	int Ndata_NPS_cal_fly_adcSampWaveform;
	int Ndata_NPS_cal_fly_block_clusterID;
	int Ndata_NPS_cal_fly_e;
	int Ndata_NPS_cal_fly_goodAdcMult;
	int Ndata_NPS_cal_fly_goodAdcPed;
	int Ndata_NPS_cal_fly_goodAdcPulseAmp;
	int Ndata_NPS_cal_fly_goodAdcPulseInt;
	int Ndata_NPS_cal_fly_goodAdcPulseIntRaw;
	int Ndata_NPS_cal_fly_goodAdcPulseTime;
	int Ndata_NPS_cal_fly_goodAdcTdcDiffTime;
	int Ndata_NPS_cal_fly_numGoodAdcHits;
	int Ndata_NPS_cal_trk_mult;
	int Ndata_NPS_cal_trk_p;
	int Ndata_NPS_cal_trk_px;
	int Ndata_NPS_cal_trk_py;
	int Ndata_NPS_cal_trk_pz;
	int Ndata_NPS_cal_trk_x;
	int Ndata_NPS_cal_trk_y;
	int Ndata_NPS_cal_vldColumn;
	int Ndata_NPS_cal_vldHiChannelMask;
	int Ndata_NPS_cal_vldLoChannelMask;
	int Ndata_NPS_cal_vldPMT;
	int Ndata_NPS_cal_vldRow;
	int Ndata_NPS_cal_vtpClusE;
	int Ndata_NPS_cal_vtpClusSize;
	int Ndata_NPS_cal_vtpClusTime;
	int Ndata_NPS_cal_vtpClusX;
	int Ndata_NPS_cal_vtpClusY;
	int Ndata_NPS_cal_vtpTrigCrate;
	int Ndata_NPS_cal_vtpTrigTime;
	int Ndata_NPS_cal_vtpTrigType0;
	int Ndata_NPS_cal_vtpTrigType1;
	int Ndata_NPS_cal_vtpTrigType2;
	int Ndata_NPS_cal_vtpTrigType3;
	int Ndata_NPS_cal_vtpTrigType4;
	int Ndata_NPS_cal_vtpTrigType5;

	// NPS branches
	std::array<double, MAX_CLUSTERS> NPS_cal_clusE;
	std::array<double, MAX_CLUSTERS> NPS_cal_clusSize;
	std::array<double, MAX_CLUSTERS> NPS_cal_clusT;
	std::array<double, MAX_CLUSTERS> NPS_cal_clusX;
	std::array<double, MAX_CLUSTERS> NPS_cal_clusY;
	std::array<double, MAX_CLUSTERS> NPS_cal_clusZ;
	std::array<double, NBLOCKS> NPS_cal_fly_adcCounter;
	std::array<double, NBLOCKS> NPS_cal_fly_adcErrorFlag;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPed;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPedRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPulseAmp;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPulseAmpRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPulseInt;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPulseIntRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPulseTime;
	std::array<double, NBLOCKS> NPS_cal_fly_adcPulseTimeRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPed;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPedRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPulseAmp;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPulseAmpRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPulseInt;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPulseIntRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPulseTime;
	std::array<double, NBLOCKS> NPS_cal_fly_adcSampPulseTimeRaw;
	std::array<double, NDATA> NPS_cal_fly_adcSampWaveform;
	std::array<double, NBLOCKS> NPS_cal_fly_block_clusterID;
	std::array<double, NBLOCKS> NPS_cal_fly_e;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcMult;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcPed;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcPulseAmp;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcPulseInt;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcPulseIntRaw;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcPulseTime;
	std::array<double, NBLOCKS> NPS_cal_fly_goodAdcTdcDiffTime;
	std::array<double, NBLOCKS> NPS_cal_fly_numGoodAdcHits;
	std::array<double, NBLOCKS> NPS_cal_trk_mult;
	std::array<double, NBLOCKS> NPS_cal_trk_p;
	std::array<double, NBLOCKS> NPS_cal_trk_px;
	std::array<double, NBLOCKS> NPS_cal_trk_py;
	std::array<double, NBLOCKS> NPS_cal_trk_pz;
	std::array<double, NBLOCKS> NPS_cal_trk_x;
	std::array<double, NBLOCKS> NPS_cal_trk_y;

	// vtp
	std::array<double, NBLOCKS> NPS_cal_vtpClusE;
	std::array<double, NBLOCKS> NPS_cal_vtpClusSize;
	std::array<double, NBLOCKS> NPS_cal_vtpClusTime;
	std::array<double, NBLOCKS> NPS_cal_vtpClusX;
	std::array<double, NBLOCKS> NPS_cal_vtpClusY;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigCrate;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigTime;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigType0;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigType1;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigType2;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigType3;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigType4;
	std::array<double, MAX_VTP_TRIGGERS> NPS_cal_vtpTrigType5;

	// Additional NPS branches
	double NPS_cal_etot;
	double NPS_cal_fly_earray;
	double NPS_cal_fly_nclust;
	double NPS_cal_fly_ntracks;
	double NPS_cal_fly_totNumAdcHits;
	double NPS_cal_fly_totNumGoodAdcHits;
	double NPS_cal_nclust;
	double NPS_cal_nhits;
	double NPS_cal_trk_vx;
	double NPS_cal_trk_vy;
	double NPS_cal_trk_vz;
	double NPS_cal_vldErrorFlag;
	double NPS_cal_vtpErrorFlag;
	double NPS_kin_secondary_Erecoil;
	double NPS_kin_secondary_MandelS;
	double NPS_kin_secondary_MandelT;
	double NPS_kin_secondary_MandelU;
	double NPS_kin_secondary_Mrecoil;
	double NPS_kin_secondary_Prec_x;
	double NPS_kin_secondary_Prec_y;
	double NPS_kin_secondary_Prec_z;
	double NPS_kin_secondary_emiss;
	double NPS_kin_secondary_emiss_nuc;
	double NPS_kin_secondary_ph_bq;
	double NPS_kin_secondary_ph_xq;
	double NPS_kin_secondary_phb_cm;
	double NPS_kin_secondary_phx_cm;
	double NPS_kin_secondary_pmiss;
	double NPS_kin_secondary_pmiss_x;
	double NPS_kin_secondary_pmiss_y;
	double NPS_kin_secondary_pmiss_z;
	double NPS_kin_secondary_px_cm;
	double NPS_kin_secondary_t_tot_cm;
	double NPS_kin_secondary_tb;
	double NPS_kin_secondary_tb_cm;
	double NPS_kin_secondary_th_bq;
	double NPS_kin_secondary_th_xq;
	double NPS_kin_secondary_thb_cm;
	double NPS_kin_secondary_thx_cm;
	double NPS_kin_secondary_tx;
	double NPS_kin_secondary_tx_cm;
	double NPS_kin_secondary_xangle;
	double NPScorrDS_measCurr;
	double NPScorrDS_setCurr;
	double NPScorrUS_measCurr;
	double NPScorrUS_setCurr;
};

void setBranchAddresses(TChain *&chain, npsBranches &buffer);
void readSignal(
	const int &NSampWaveForm, const std::array<double, NPS::NDATA> &SampWaveForm, std::vector<int> &blocks,
	std::vector<std::vector<double>> &signals
);

} // namespace NPS

#endif