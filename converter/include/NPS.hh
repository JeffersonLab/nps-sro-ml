#ifndef NPS_HH
#define NPS_HH

#include "TChain.h"
#include <array>
#include <iostream>
#include <string>

const int NTIME = 110;						// number of time samples for each fADC channel
const int NCOLS = 30;						// number of calo colomns tested in this run (3 for runs 55 and 56)
const int NROWS = 36;						// number of calo blocks in each colomn
const int NBLOCKS = NCOLS * NROWS;			// number of tested calo blocks
const double dt = 4.;						// time bin (sample) width (4 ns), the total time window is
											// then NTIME*dt
const int NSLOTS = 1104;					// nb maximal de slots dans tous les fADC
const int NDATA = NSLOTS * (NTIME + 1 + 1); //(1104 slots fADC au total mais pas tous utilises y compris 2 PM

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
	std::array<double, 256> NPS_cal_clusE;
	std::array<double, 256> NPS_cal_clusSize;
	std::array<double, 256> NPS_cal_clusT;
	std::array<double, 256> NPS_cal_clusX;
	std::array<double, 256> NPS_cal_clusY;
	std::array<double, 256> NPS_cal_clusZ;
	std::array<double, 1024> NPS_cal_fly_adcCounter;
	std::array<double, 1024> NPS_cal_fly_adcErrorFlag;
	std::array<double, 1024> NPS_cal_fly_adcPed;
	std::array<double, 1024> NPS_cal_fly_adcPedRaw;
	std::array<double, 1024> NPS_cal_fly_adcPulseAmp;
	std::array<double, 1024> NPS_cal_fly_adcPulseAmpRaw;
	std::array<double, 1024> NPS_cal_fly_adcPulseInt;
	std::array<double, 1024> NPS_cal_fly_adcPulseIntRaw;
	std::array<double, 1024> NPS_cal_fly_adcPulseTime;
	std::array<double, 1024> NPS_cal_fly_adcPulseTimeRaw;
	std::array<double, 1024> NPS_cal_fly_adcSampPed;
	std::array<double, 1024> NPS_cal_fly_adcSampPedRaw;
	std::array<double, 1024> NPS_cal_fly_adcSampPulseAmp;
	std::array<double, 1024> NPS_cal_fly_adcSampPulseAmpRaw;
	std::array<double, 1024> NPS_cal_fly_adcSampPulseInt;
	std::array<double, 1024> NPS_cal_fly_adcSampPulseIntRaw;
	std::array<double, 1024> NPS_cal_fly_adcSampPulseTime;
	std::array<double, 1024> NPS_cal_fly_adcSampPulseTimeRaw;
	std::array<double, 112 * 1104> NPS_cal_fly_adcSampWaveform;
	std::array<double, 1080> NPS_cal_fly_block_clusterID;
	std::array<double, 1080> NPS_cal_fly_e;
	std::array<double, 1080> NPS_cal_fly_goodAdcMult;
	std::array<double, 1080> NPS_cal_fly_goodAdcPed;
	std::array<double, 1080> NPS_cal_fly_goodAdcPulseAmp;
	std::array<double, 1080> NPS_cal_fly_goodAdcPulseInt;
	std::array<double, 1080> NPS_cal_fly_goodAdcPulseIntRaw;
	std::array<double, 1080> NPS_cal_fly_goodAdcPulseTime;
	std::array<double, 1080> NPS_cal_fly_goodAdcTdcDiffTime;
	std::array<double, 1080> NPS_cal_fly_numGoodAdcHits;
	std::array<double, 256> NPS_cal_trk_mult;
	std::array<double, 256> NPS_cal_trk_p;
	std::array<double, 256> NPS_cal_trk_px;
	std::array<double, 256> NPS_cal_trk_py;
	std::array<double, 256> NPS_cal_trk_pz;
	std::array<double, 256> NPS_cal_trk_x;
	std::array<double, 256> NPS_cal_trk_y;

	// vld
	std::array<double, 256> NPS_cal_vldColumn;
	std::array<double, 256> NPS_cal_vldHiChannelMask;
	std::array<double, 256> NPS_cal_vldLoChannelMask;
	std::array<double, 256> NPS_cal_vldPMT;
	std::array<double, 256> NPS_cal_vldRow;

	// vtp
	std::array<double, 2048> NPS_cal_vtpClusE;
	std::array<double, 2048> NPS_cal_vtpClusSize;
	std::array<double, 2048> NPS_cal_vtpClusTime;
	std::array<double, 2048> NPS_cal_vtpClusX;
	std::array<double, 2048> NPS_cal_vtpClusY;
	std::array<double, 256> NPS_cal_vtpTrigCrate;
	std::array<double, 256> NPS_cal_vtpTrigTime;
	std::array<double, 256> NPS_cal_vtpTrigType0;
	std::array<double, 256> NPS_cal_vtpTrigType1;
	std::array<double, 256> NPS_cal_vtpTrigType2;
	std::array<double, 256> NPS_cal_vtpTrigType3;
	std::array<double, 256> NPS_cal_vtpTrigType4;
	std::array<double, 256> NPS_cal_vtpTrigType5;

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
#endif