#include "Graph.hh"
#include "NPS.hh"
#include "TChain.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TString.h"
#include "Utilities.hh"
#include "VTP.hh"
#include "fADC250.hh"

#include <array>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Undefine the ROOT macro that conflicts with torch
#ifdef ClassDef
#undef ClassDef
#endif

#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "argparse/argparse.hpp"

void Addarguments(int argc, char **argv);
argparse::ArgumentParser ARGS("VTP Reconstruction", "1.0");

void saveGraph(const GraphUtils::GraphData &graphData, const std::string &output_file);
bool isCorruptSignals(const std::vector<std::vector<double>> &signals);
vtp_reco_evt buildVtpEventFromBuffer(NPS::npsBranches &buffer, NPS::Geometry &geometry);
int buildTargetEdges(
	GraphUtils::GraphBuilder &graphBuilder, const vtp_reco_evt &recoEvent, const vtp_reco_evt &vtpRawEvent,
	double energyDiff, double timeLow, double timeHigh
);
int main(int argc, char **argv) {
	Addarguments(argc, argv);

	const int readEntries = ARGS.get<int>("--n-events");
	const int startEntry = ARGS.get<int>("--start-event");
	const auto inputFiles = ARGS.get<std::vector<std::string>>("--input-files");
	const std::string outputDir = ARGS.get<std::string>("--output-dir");
	const std::string treeName = ARGS.get<std::string>("--tree-name");
	const std::string vmeConfig = ARGS.get<std::string>("--vme-config");
	const std::string vtpConfig = ARGS.get<std::string>("--vtp-config");
	const std::string geoConfig = ARGS.get<std::string>("--geo-config");
	const double energyDiff = ARGS.get<double>("--energy-diff");
	const std::vector<double> timeWindow = ARGS.get<std::vector<double>>("--time-window");
	const bool debug = ARGS.get<bool>("--debug");

	auto chain = new TChain(treeName.c_str());
	for (const auto &filename : inputFiles) {
		chain->Add(filename.c_str());
	}
	auto entries = chain->GetEntries();

	NPS::npsBranches buffer;
	NPS::setBranchAddresses(chain, buffer);

	int processedEntries = 0;
	int savedEvents = 0;
	int useEntries = readEntries == -1 ? chain->GetEntries() : std::min(readEntries, (int)chain->GetEntries());

	fADC250 fadcDevice(NPS::NBLOCKS, vmeConfig);
	VTP vtpDevice(NPS::NBLOCKS, NPS::NTIME, NPS::DELTA_T, vtpConfig);
	NPS::Geometry geometry(geoConfig);
	GraphUtils::GraphBuilder graphBuilder(NPS::NTIME, 0, 0, false, true);

	auto finishEvent = [&]() {
		fadcDevice.resetEvent();
		vtpDevice.resetEvent();
		graphBuilder.reset();
		processedEntries++;
	};

	while (processedEntries < useEntries) {

		// std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::endl;
		int currEvent = startEntry + processedEntries;
		chain->GetEntry(currEvent);
		std::vector<std::vector<double>> signals; // [nblocks][ntime]
		std::vector<int> blocks;				  // [nblocks]

		auto signalFlag = NPS::readSignal(
			buffer.Ndata_NPS_cal_fly_adcSampWaveform, buffer.NPS_cal_fly_adcSampWaveform, blocks, signals
		);

		if (signalFlag != 0) {
			auto msg = Form("readSignal returned error code %d", signalFlag);
			if (debug) {
				std::cerr << msg << " in event " << currEvent << std::endl;
			}
			finishEvent();
			continue;
		}
		if (isCorruptSignals(signals)) {
			auto msg = Form(
				"Corrupt signals with size [%zu][%zu] in event %d", signals.size(),
				signals.size() > 0 ? signals[0].size() : 0, currEvent
			);
			if (debug) {
				std::cerr << msg << " in event " << currEvent << std::endl;
			}
			finishEvent();
			continue;
		}
		graphBuilder.addNodes(blocks, signals);

		for (int i = 0; i < blocks.size(); i++) {
			auto channel = blocks[i];
			fadcDevice.processRawWaveform(signals[i], channel, 0);
		}
		auto fadc250_evt = fadcDevice.getEvent(); // to make sure mEvent is updated
		auto nhits = fadc250_evt.nhits;

		for (int i = 0; i < nhits; i++) {
			// use vtpDevice to decide vtp cluster membership

			auto seedChannel = fadc250_evt.channels[i];
			auto seedTime = fadc250_evt.times[i];
			auto seedEnergy = fadc250_evt.energies[i];

			std::vector<int> gridTimes;		  // [num_blocks_in_grid]
			std::vector<double> gridEnergies; // [num_blocks_in_grid]
			std::vector<int> gridChannels;	  // [num_blocks_in_grid]

			for (int j = 0; j < nhits; j++) {
				if (i == j) {
					continue;
				}
				auto ch = fadc250_evt.channels[j];
				auto e = fadc250_evt.energies[j];
				auto t = fadc250_evt.times[j];
				if (geometry.isInsideGrid(seedChannel, ch, 3)) {
					gridTimes.push_back(t);
					gridEnergies.push_back(e);
					gridChannels.push_back(ch);
				}
			}

			vtpDevice.process(seedChannel, seedTime, seedEnergy, gridChannels, gridTimes, gridEnergies);
		}

		auto recoEvent = vtpDevice.getEvent();
		auto vtpEvent = buildVtpEventFromBuffer(buffer, geometry);
		auto nClust = buildTargetEdges(graphBuilder, recoEvent, vtpEvent, energyDiff, timeWindow[0], timeWindow[1]);

		// Build graph and save tensors
		auto graphData = graphBuilder.buildGraph();
		auto components = GraphUtils::getConnectedComponents(graphData.edgeTargetIndex);
		assert(components.size() == nClust);
		saveGraph(graphData, Form("%s/%08d.pt", outputDir.c_str(), savedEvents));
		finishEvent();
		savedEvents++;
	}

	// Final report
	std::cout << std::endl;
	std::cout << "Processed " << processedEntries << " events, saved " << savedEvents << " events." << std::endl;
	return 0;
}

void saveGraph(const GraphUtils::GraphData &graphData, const std::string &outputFile) {

	auto nodeIds = TorchUtils::toTensor(graphData.nodeIds);				// [num_nodes]
	auto nodeFeatures = TorchUtils::toTensor2D(graphData.nodeFeatures); // [num_nodes][num_node_features]
	auto nodeTargets = TorchUtils::toTensor2D(graphData.nodeTargets);	// [num_nodes][num_targets]
	auto edgeFeatures = TorchUtils::toTensor2D(graphData.edgeFeatures); // [num_edges][num_edge_features]
	auto edgeTargetFeatures =
		TorchUtils::toTensor2D(graphData.edgeTargetFeatures);	  // [num_target_edges][num_target_edge_features]
	auto edgeIndex = TorchUtils::toTensor2D(graphData.edgeIndex); // [2][num_edges]
	auto edgeTargetIndex = TorchUtils::toTensor2D(graphData.edgeTargetIndex); // [2][num_target_edges]

	std::filesystem::create_directories(std::filesystem::path(outputFile).parent_path());
	TorchUtils::saveTensors(
		outputFile, nodeIds, nodeFeatures, nodeTargets, edgeFeatures, edgeTargetFeatures, edgeIndex, edgeTargetIndex
	);
}

bool isCorruptSignals(const std::vector<std::vector<double>> &signals) {
	bool corrupt = false;
	corrupt |= signals.size() == 0;
	corrupt |= signals.size() > 0 && signals[0].size() != NPS::NTIME;
	return corrupt;
}

vtp_reco_evt buildVtpEventFromBuffer(NPS::npsBranches &buffer, NPS::Geometry &geometry) {

	vtp_reco_evt evt{
		.nseeds = 0,
		.clus_sizes = {},
		.channels = {},
		.times = {},
		.energies = {},
	};

	evt.nseeds = buffer.Ndata_NPS_cal_vtpClusX;
	for (int iclus = 0; iclus < evt.nseeds; iclus++) {
		auto col = buffer.NPS_cal_vtpClusX[iclus];
		auto row = buffer.NPS_cal_vtpClusY[iclus];
		auto ch = geometry.getBlockFromColRow(col, row);
		auto e = buffer.NPS_cal_vtpClusE[iclus];
		auto t = buffer.NPS_cal_vtpClusTime[iclus];
		auto size = buffer.NPS_cal_vtpClusSize[iclus];

		evt.clus_sizes.push_back(size);
		evt.channels.push_back({ch});
		evt.times.push_back({static_cast<int>(t)});
		evt.energies.push_back({e});
	}
	return evt;
}

int buildTargetEdges(
	GraphUtils::GraphBuilder &graphBuilder, const vtp_reco_evt &recoEvent, const vtp_reco_evt &vtpRawEvent,
	double energyDiff, double timeLow, double timeHigh
) {

	auto sum = [](const std::vector<double> &vec) { return std::accumulate(vec.begin(), vec.end(), 0.0); };

	std::unordered_set<int> usedRecoIndices;

	for (int iclus = 0; iclus < vtpRawEvent.nseeds; iclus++) {
		auto vtp_ch = vtpRawEvent.channels[iclus][0];  // vtp seed channel
		auto vtp_e = sum(vtpRawEvent.energies[iclus]); // vtp cluster energy
		auto vtp_time = vtpRawEvent.times[iclus][0];   // vtp seed time
		auto vtp_size = vtpRawEvent.clus_sizes[iclus]; // vtp cluster size

		for (int i_reco = 0; i_reco < recoEvent.nseeds; i_reco++) {

			if (usedRecoIndices.count(i_reco)) {
				continue;
			}

			auto reco_ch = recoEvent.channels[i_reco][0];  // reco seed channel
			auto reco_e = sum(recoEvent.energies[i_reco]); // reco cluster energy
			auto reco_time = recoEvent.times[i_reco][0];   // reco seed time
			auto reco_size = recoEvent.clus_sizes[i_reco]; // reco cluster size

			bool match = (vtp_ch == reco_ch);
			match &= (vtp_time == reco_time);
			match &= (vtp_size == reco_size);
			match &= (std::abs(vtp_e - reco_e) < energyDiff);

			bool time_cut = (vtp_time >= timeLow) && (vtp_time <= timeHigh);

			if (match) {
				usedRecoIndices.insert(i_reco);
				if (time_cut) {
					continue;
				}

				for (const auto &ch : recoEvent.channels[i_reco]) {
					if (ch != reco_ch) {
						graphBuilder.addEdgeTarget(reco_ch, ch);
					}
				}
			}
		}
	}
	return usedRecoIndices.size();
}

void Addarguments(int argc, char **argv) {

	ARGS.add_argument("-i", "--input-files")
		.nargs(argparse::nargs_pattern::at_least_one)
		.help("input root files")
		.default_value(std::vector<std::string>{"./cache/nps_hms_coin_4454_0_1_-1.root"})
		.required();

	ARGS.add_argument("-o", "--output-dir").help("output directory").default_value(std::string("./output")).required();

	ARGS.add_argument("-t", "--tree-name")
		.help("name of the tree in the input file")
		.default_value(std::string("T"))
		.required();

	ARGS.add_argument("-n", "--n-events")
		.help("number of events to process, -1 for all")
		.default_value(-1)
		.scan<'i', int>();

	ARGS.add_argument("--start-event").help("starting event number").default_value(0).scan<'i', int>();

	ARGS.add_argument("--vme-config")
		.default_value(std::string("/u/group/c-kaonlt/USERS/ckin/nps-sro-ml/temp/nps_run_4599_vme_config.csv"))
		.help("VME config file")
		.required();

	ARGS.add_argument("--vtp-config")
		.default_value(std::string("/u/group/c-kaonlt/USERS/ckin/nps-sro-ml/temp/nps_run_4599_vtp_config.csv"))
		.help("VTP config file")
		.required();

	ARGS.add_argument("--geo-config")
		.default_value(std::string("database/channel_map.csv"))
		.help("NPS Geometry config file")
		.required();

	ARGS.add_argument("--energy-diff")
		.help("time window for VTP clustering")
		.default_value(5.0)
		.scan<'g', double>()
		.required();

	ARGS.add_argument("--time-window")
		.help("time window which VTP clusters are matched to reconstructed clusters [ns]")
		.default_value(std::vector<double>({15.0, 93.0}))
		.scan<'g', double>()
		.required();

	ARGS.add_argument("-d", "--debug").help("debug mode").flag();
	try {
		ARGS.parse_args(argc, argv);
	} catch (const std::runtime_error &err) {
		std::cout << err.what() << std::endl;
		std::cout << ARGS;
		exit(0);
	}
}