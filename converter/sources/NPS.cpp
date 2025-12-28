#include "NPS.hh"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace NPS {

Geometry::Geometry(const std::string &config_file) { loadConfig(config_file); }

void Geometry::loadConfig(const std::string &config_file) {
	std::ifstream fin(config_file);
	if (!fin.is_open()) {
		throw std::runtime_error("Cannot open Geometry config: " + config_file);
	}

	std::string line;
	while (std::getline(fin, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}

		BlockInfo info;
		std::istringstream iss(line);
		iss >> info.channel >> info.row >> info.col >> info.crate >> info.slot;
		if (!iss) {
			continue;
		}
		mBlocks.push_back(info);
		mIndexMap[info.channel] = info;
		mRcToIndex[{info.row, info.col}] = info.channel;
	}
}

int Geometry::getBlockFromColRow(int col, int row) const {
	auto it = mRcToIndex.find({row, col});
	if (it == mRcToIndex.end())
		throw std::out_of_range("Invalid (row, col)");
	return it->second;
}

std::pair<int, int> Geometry::getColRowFromBlock(int block) const {
	auto it = mIndexMap.find(block);
	if (it == mIndexMap.end())
		throw std::out_of_range("Invalid block index");
	return {it->second.col, it->second.row};
}

int Geometry::getCrateFromBlock(int block) const {
	auto it = mIndexMap.find(block);
	if (it == mIndexMap.end())
		throw std::out_of_range("Invalid block index");
	return it->second.crate;
}

int Geometry::getSlotFromBlock(int block) const {
	auto it = mIndexMap.find(block);
	if (it == mIndexMap.end())
		throw std::out_of_range("Invalid block index");
	return it->second.slot;
}

bool Geometry::isNeighbour(int ch1, int ch2) const {
	auto [col1, row1] = getColRowFromBlock(ch1);
	auto [col2, row2] = getColRowFromBlock(ch2);
	return (std::abs(col1 - col2) <= 1) && (std::abs(row1 - row2) <= 1) && !(col1 == col2 && row1 == row2);
}

bool Geometry::isInsideGrid(int seedChannel, int channel, int gridSize) const {
	auto [seedCol, seedRow] = getColRowFromBlock(seedChannel);
	auto [chCol, chRow] = getColRowFromBlock(channel);
	int halfGrid = gridSize / 2;
	return (std::abs(seedCol - chCol) <= halfGrid) && (std::abs(seedRow - chRow) <= halfGrid);
}

void setBranchAddresses(TChain *&chain, npsBranches &buffer) {

	// Global branches
	chain->SetBranchAddress("g.evtime", &buffer.g_evtime);

	// Ndata.NPS branches
	chain->SetBranchAddress("Ndata.NPS.cal.clusE", &buffer.Ndata_NPS_cal_clusE);
	chain->SetBranchAddress("Ndata.NPS.cal.clusSize", &buffer.Ndata_NPS_cal_clusSize);
	chain->SetBranchAddress("Ndata.NPS.cal.clusT", &buffer.Ndata_NPS_cal_clusT);
	chain->SetBranchAddress("Ndata.NPS.cal.clusX", &buffer.Ndata_NPS_cal_clusX);
	chain->SetBranchAddress("Ndata.NPS.cal.clusY", &buffer.Ndata_NPS_cal_clusY);
	chain->SetBranchAddress("Ndata.NPS.cal.clusZ", &buffer.Ndata_NPS_cal_clusZ);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcCounter", &buffer.Ndata_NPS_cal_fly_adcCounter);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcErrorFlag", &buffer.Ndata_NPS_cal_fly_adcErrorFlag);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPed", &buffer.Ndata_NPS_cal_fly_adcPed);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPedRaw", &buffer.Ndata_NPS_cal_fly_adcPedRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPulseAmp", &buffer.Ndata_NPS_cal_fly_adcPulseAmp);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPulseAmpRaw", &buffer.Ndata_NPS_cal_fly_adcPulseAmpRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPulseInt", &buffer.Ndata_NPS_cal_fly_adcPulseInt);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPulseIntRaw", &buffer.Ndata_NPS_cal_fly_adcPulseIntRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPulseTime", &buffer.Ndata_NPS_cal_fly_adcPulseTime);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcPulseTimeRaw", &buffer.Ndata_NPS_cal_fly_adcPulseTimeRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPed", &buffer.Ndata_NPS_cal_fly_adcSampPed);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPedRaw", &buffer.Ndata_NPS_cal_fly_adcSampPedRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPulseAmp", &buffer.Ndata_NPS_cal_fly_adcSampPulseAmp);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPulseAmpRaw", &buffer.Ndata_NPS_cal_fly_adcSampPulseAmpRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPulseInt", &buffer.Ndata_NPS_cal_fly_adcSampPulseInt);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPulseIntRaw", &buffer.Ndata_NPS_cal_fly_adcSampPulseIntRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPulseTime", &buffer.Ndata_NPS_cal_fly_adcSampPulseTime);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampPulseTimeRaw", &buffer.Ndata_NPS_cal_fly_adcSampPulseTimeRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.adcSampWaveform", &buffer.Ndata_NPS_cal_fly_adcSampWaveform);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.block_clusterID", &buffer.Ndata_NPS_cal_fly_block_clusterID);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.e", &buffer.Ndata_NPS_cal_fly_e);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcMult", &buffer.Ndata_NPS_cal_fly_goodAdcMult);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcPed", &buffer.Ndata_NPS_cal_fly_goodAdcPed);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcPulseAmp", &buffer.Ndata_NPS_cal_fly_goodAdcPulseAmp);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcPulseInt", &buffer.Ndata_NPS_cal_fly_goodAdcPulseInt);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcPulseIntRaw", &buffer.Ndata_NPS_cal_fly_goodAdcPulseIntRaw);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcPulseTime", &buffer.Ndata_NPS_cal_fly_goodAdcPulseTime);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.goodAdcTdcDiffTime", &buffer.Ndata_NPS_cal_fly_goodAdcTdcDiffTime);
	chain->SetBranchAddress("Ndata.NPS.cal.fly.numGoodAdcHits", &buffer.Ndata_NPS_cal_fly_numGoodAdcHits);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.mult", &buffer.Ndata_NPS_cal_trk_mult);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.p", &buffer.Ndata_NPS_cal_trk_p);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.px", &buffer.Ndata_NPS_cal_trk_px);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.py", &buffer.Ndata_NPS_cal_trk_py);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.pz", &buffer.Ndata_NPS_cal_trk_pz);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.x", &buffer.Ndata_NPS_cal_trk_x);
	chain->SetBranchAddress("Ndata.NPS.cal.trk.y", &buffer.Ndata_NPS_cal_trk_y);
	chain->SetBranchAddress("Ndata.NPS.cal.vldColumn", &buffer.Ndata_NPS_cal_vldColumn);
	chain->SetBranchAddress("Ndata.NPS.cal.vldHiChannelMask", &buffer.Ndata_NPS_cal_vldHiChannelMask);
	chain->SetBranchAddress("Ndata.NPS.cal.vldLoChannelMask", &buffer.Ndata_NPS_cal_vldLoChannelMask);
	chain->SetBranchAddress("Ndata.NPS.cal.vldPMT", &buffer.Ndata_NPS_cal_vldPMT);
	chain->SetBranchAddress("Ndata.NPS.cal.vldRow", &buffer.Ndata_NPS_cal_vldRow);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpClusE", &buffer.Ndata_NPS_cal_vtpClusE);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpClusSize", &buffer.Ndata_NPS_cal_vtpClusSize);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpClusTime", &buffer.Ndata_NPS_cal_vtpClusTime);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpClusX", &buffer.Ndata_NPS_cal_vtpClusX);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpClusY", &buffer.Ndata_NPS_cal_vtpClusY);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigCrate", &buffer.Ndata_NPS_cal_vtpTrigCrate);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigTime", &buffer.Ndata_NPS_cal_vtpTrigTime);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigType0", &buffer.Ndata_NPS_cal_vtpTrigType0);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigType1", &buffer.Ndata_NPS_cal_vtpTrigType1);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigType2", &buffer.Ndata_NPS_cal_vtpTrigType2);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigType3", &buffer.Ndata_NPS_cal_vtpTrigType3);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigType4", &buffer.Ndata_NPS_cal_vtpTrigType4);
	chain->SetBranchAddress("Ndata.NPS.cal.vtpTrigType5", &buffer.Ndata_NPS_cal_vtpTrigType5);

	// NPS branches
	chain->SetBranchAddress("NPS.cal.clusE", &buffer.NPS_cal_clusE[0]);
	chain->SetBranchAddress("NPS.cal.clusSize", &buffer.NPS_cal_clusSize[0]);
	chain->SetBranchAddress("NPS.cal.clusT", &buffer.NPS_cal_clusT[0]);
	chain->SetBranchAddress("NPS.cal.clusX", &buffer.NPS_cal_clusX[0]);
	chain->SetBranchAddress("NPS.cal.clusY", &buffer.NPS_cal_clusY[0]);
	chain->SetBranchAddress("NPS.cal.clusZ", &buffer.NPS_cal_clusZ[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcCounter", &buffer.NPS_cal_fly_adcCounter[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcErrorFlag", &buffer.NPS_cal_fly_adcErrorFlag[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPed", &buffer.NPS_cal_fly_adcPed[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPedRaw", &buffer.NPS_cal_fly_adcPedRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPulseAmp", &buffer.NPS_cal_fly_adcPulseAmp[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPulseAmpRaw", &buffer.NPS_cal_fly_adcPulseAmpRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPulseInt", &buffer.NPS_cal_fly_adcPulseInt[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPulseIntRaw", &buffer.NPS_cal_fly_adcPulseIntRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPulseTime", &buffer.NPS_cal_fly_adcPulseTime[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcPulseTimeRaw", &buffer.NPS_cal_fly_adcPulseTimeRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPed", &buffer.NPS_cal_fly_adcSampPed[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPedRaw", &buffer.NPS_cal_fly_adcSampPedRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPulseAmp", &buffer.NPS_cal_fly_adcSampPulseAmp[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPulseAmpRaw", &buffer.NPS_cal_fly_adcSampPulseAmpRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPulseInt", &buffer.NPS_cal_fly_adcSampPulseInt[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPulseIntRaw", &buffer.NPS_cal_fly_adcSampPulseIntRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPulseTime", &buffer.NPS_cal_fly_adcSampPulseTime[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampPulseTimeRaw", &buffer.NPS_cal_fly_adcSampPulseTimeRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.adcSampWaveform", &buffer.NPS_cal_fly_adcSampWaveform[0]);
	chain->SetBranchAddress("NPS.cal.fly.block_clusterID", &buffer.NPS_cal_fly_block_clusterID[0]);
	chain->SetBranchAddress("NPS.cal.fly.e", &buffer.NPS_cal_fly_e[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcMult", &buffer.NPS_cal_fly_goodAdcMult[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcPed", &buffer.NPS_cal_fly_goodAdcPed[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcPulseAmp", &buffer.NPS_cal_fly_goodAdcPulseAmp[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcPulseInt", &buffer.NPS_cal_fly_goodAdcPulseInt[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcPulseIntRaw", &buffer.NPS_cal_fly_goodAdcPulseIntRaw[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcPulseTime", &buffer.NPS_cal_fly_goodAdcPulseTime[0]);
	chain->SetBranchAddress("NPS.cal.fly.goodAdcTdcDiffTime", &buffer.NPS_cal_fly_goodAdcTdcDiffTime[0]);
	chain->SetBranchAddress("NPS.cal.fly.numGoodAdcHits", &buffer.NPS_cal_fly_numGoodAdcHits[0]);
	chain->SetBranchAddress("NPS.cal.trk.mult", &buffer.NPS_cal_trk_mult[0]);
	chain->SetBranchAddress("NPS.cal.trk.p", &buffer.NPS_cal_trk_p[0]);
	chain->SetBranchAddress("NPS.cal.trk.px", &buffer.NPS_cal_trk_px[0]);
	chain->SetBranchAddress("NPS.cal.trk.py", &buffer.NPS_cal_trk_py[0]);
	chain->SetBranchAddress("NPS.cal.trk.pz", &buffer.NPS_cal_trk_pz[0]);
	chain->SetBranchAddress("NPS.cal.trk.x", &buffer.NPS_cal_trk_x[0]);
	chain->SetBranchAddress("NPS.cal.trk.y", &buffer.NPS_cal_trk_y[0]);
	chain->SetBranchAddress("NPS.cal.vtpClusE", &buffer.NPS_cal_vtpClusE[0]);
	chain->SetBranchAddress("NPS.cal.vtpClusSize", &buffer.NPS_cal_vtpClusSize[0]);
	chain->SetBranchAddress("NPS.cal.vtpClusTime", &buffer.NPS_cal_vtpClusTime[0]);
	chain->SetBranchAddress("NPS.cal.vtpClusX", &buffer.NPS_cal_vtpClusX[0]);
	chain->SetBranchAddress("NPS.cal.vtpClusY", &buffer.NPS_cal_vtpClusY[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigCrate", &buffer.NPS_cal_vtpTrigCrate[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigTime", &buffer.NPS_cal_vtpTrigTime[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigType0", &buffer.NPS_cal_vtpTrigType0[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigType1", &buffer.NPS_cal_vtpTrigType1[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigType2", &buffer.NPS_cal_vtpTrigType2[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigType3", &buffer.NPS_cal_vtpTrigType3[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigType4", &buffer.NPS_cal_vtpTrigType4[0]);
	chain->SetBranchAddress("NPS.cal.vtpTrigType5", &buffer.NPS_cal_vtpTrigType5[0]);
	chain->SetBranchAddress("NPS.cal.etot", &buffer.NPS_cal_etot);
	chain->SetBranchAddress("NPS.cal.fly.earray", &buffer.NPS_cal_fly_earray);
	chain->SetBranchAddress("NPS.cal.fly.nclust", &buffer.NPS_cal_fly_nclust);
	chain->SetBranchAddress("NPS.cal.fly.ntracks", &buffer.NPS_cal_fly_ntracks);
	chain->SetBranchAddress("NPS.cal.fly.totNumAdcHits", &buffer.NPS_cal_fly_totNumAdcHits);
	chain->SetBranchAddress("NPS.cal.fly.totNumGoodAdcHits", &buffer.NPS_cal_fly_totNumGoodAdcHits);
	chain->SetBranchAddress("NPS.cal.nclust", &buffer.NPS_cal_nclust);
	chain->SetBranchAddress("NPS.cal.nhits", &buffer.NPS_cal_nhits);
	chain->SetBranchAddress("NPS.cal.trk.vx", &buffer.NPS_cal_trk_vx);
	chain->SetBranchAddress("NPS.cal.trk.vy", &buffer.NPS_cal_trk_vy);
	chain->SetBranchAddress("NPS.cal.trk.vz", &buffer.NPS_cal_trk_vz);
	chain->SetBranchAddress("NPS.cal.vldErrorFlag", &buffer.NPS_cal_vldErrorFlag);
	chain->SetBranchAddress("NPS.cal.vtpErrorFlag", &buffer.NPS_cal_vtpErrorFlag);
	chain->SetBranchAddress("NPS.kin.secondary.Erecoil", &buffer.NPS_kin_secondary_Erecoil);
	chain->SetBranchAddress("NPS.kin.secondary.MandelS", &buffer.NPS_kin_secondary_MandelS);
	chain->SetBranchAddress("NPS.kin.secondary.MandelT", &buffer.NPS_kin_secondary_MandelT);
	chain->SetBranchAddress("NPS.kin.secondary.MandelU", &buffer.NPS_kin_secondary_MandelU);
	chain->SetBranchAddress("NPS.kin.secondary.Mrecoil", &buffer.NPS_kin_secondary_Mrecoil);
	chain->SetBranchAddress("NPS.kin.secondary.Prec_x", &buffer.NPS_kin_secondary_Prec_x);
	chain->SetBranchAddress("NPS.kin.secondary.Prec_y", &buffer.NPS_kin_secondary_Prec_y);
	chain->SetBranchAddress("NPS.kin.secondary.Prec_z", &buffer.NPS_kin_secondary_Prec_z);
	chain->SetBranchAddress("NPS.kin.secondary.emiss", &buffer.NPS_kin_secondary_emiss);
	chain->SetBranchAddress("NPS.kin.secondary.emiss_nuc", &buffer.NPS_kin_secondary_emiss_nuc);
	chain->SetBranchAddress("NPS.kin.secondary.ph_bq", &buffer.NPS_kin_secondary_ph_bq);
	chain->SetBranchAddress("NPS.kin.secondary.ph_xq", &buffer.NPS_kin_secondary_ph_xq);
	chain->SetBranchAddress("NPS.kin.secondary.phb_cm", &buffer.NPS_kin_secondary_phb_cm);
	chain->SetBranchAddress("NPS.kin.secondary.phx_cm", &buffer.NPS_kin_secondary_phx_cm);
	chain->SetBranchAddress("NPS.kin.secondary.pmiss", &buffer.NPS_kin_secondary_pmiss);
	chain->SetBranchAddress("NPS.kin.secondary.pmiss_x", &buffer.NPS_kin_secondary_pmiss_x);
	chain->SetBranchAddress("NPS.kin.secondary.pmiss_y", &buffer.NPS_kin_secondary_pmiss_y);
	chain->SetBranchAddress("NPS.kin.secondary.pmiss_z", &buffer.NPS_kin_secondary_pmiss_z);
	chain->SetBranchAddress("NPS.kin.secondary.px_cm", &buffer.NPS_kin_secondary_px_cm);
	chain->SetBranchAddress("NPS.kin.secondary.t_tot_cm", &buffer.NPS_kin_secondary_t_tot_cm);
	chain->SetBranchAddress("NPS.kin.secondary.tb", &buffer.NPS_kin_secondary_tb);
	chain->SetBranchAddress("NPS.kin.secondary.tb_cm", &buffer.NPS_kin_secondary_tb_cm);
	chain->SetBranchAddress("NPS.kin.secondary.th_bq", &buffer.NPS_kin_secondary_th_bq);
	chain->SetBranchAddress("NPS.kin.secondary.th_xq", &buffer.NPS_kin_secondary_th_xq);
	chain->SetBranchAddress("NPS.kin.secondary.thb_cm", &buffer.NPS_kin_secondary_thb_cm);
	chain->SetBranchAddress("NPS.kin.secondary.thx_cm", &buffer.NPS_kin_secondary_thx_cm);
	chain->SetBranchAddress("NPS.kin.secondary.tx", &buffer.NPS_kin_secondary_tx);
	chain->SetBranchAddress("NPS.kin.secondary.tx_cm", &buffer.NPS_kin_secondary_tx_cm);
	chain->SetBranchAddress("NPS.kin.secondary.xangle", &buffer.NPS_kin_secondary_xangle);
	chain->SetBranchAddress("NPScorrDS_measCurr", &buffer.NPScorrDS_measCurr);
	chain->SetBranchAddress("NPScorrDS_setCurr", &buffer.NPScorrDS_setCurr);
	chain->SetBranchAddress("NPScorrUS_measCurr", &buffer.NPScorrUS_measCurr);
	chain->SetBranchAddress("NPScorrUS_setCurr", &buffer.NPScorrUS_setCurr);
}

void readSignal(
	const int &NSampWaveForm, const std::array<double, NPS::NDATA> &SampWaveForm, std::vector<int> &blocks,
	std::vector<std::vector<double>> &signals
) {
	signals.clear();
	blocks.clear();

	// Reserve memory to minimize reallocations
	blocks.reserve(NPS::NBLOCKS);
	signals.reserve(NPS::NBLOCKS);

	std::unordered_set<int> block_seen;
	block_seen.reserve(NPS::NBLOCKS);

	int ns = 0;

	while (ns < NSampWaveForm) {
		int bloc = static_cast<int>(SampWaveForm[ns++]);  // block (slot) number
		int nsamp = static_cast<int>(SampWaveForm[ns++]); // number of samples (e.g., 110)

		// Check bounds — ensure enough samples left
		if (ns + nsamp > NSampWaveForm) {
			std::cerr << "Warning: not enough samples for block " << bloc << " (expected " << nsamp << ", available "
					  << (NSampWaveForm - ns) << "). Stopping readSignal.\n";
			break;
		}

		// Historical remapping
		if (bloc == 2000) {
			bloc = 1080;
		} else if (bloc == 2001) {
			bloc = 1081;
		}

		// Skip invalid or duplicate blocks
		if (bloc < 0 || bloc >= NPS::NBLOCKS || block_seen.find(bloc) != block_seen.end()) {
			ns += nsamp;
			continue;
		}

		block_seen.insert(bloc);
		blocks.emplace_back(bloc);

		// Read waveform samples efficiently
		std::vector<double> sig;
		sig.reserve(nsamp);

		for (int it = 0; it < nsamp; ++it)
			sig.emplace_back(SampWaveForm[ns++]);

		// Move the waveform into the result vector (no copy)
		signals.emplace_back(std::move(sig));
	}
}

} // namespace NPS
