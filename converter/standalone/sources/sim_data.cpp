#include "Graph.hh"
#include "NPS.hh"
#include "TChain.h"
#include "TFile.h"
#include "TString.h"
#include "Utilities.hh"

#include <array>
#include <cassert>
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

void addArguments(int argc, char **argv);
argparse::ArgumentParser ARGS("Simulation_Data", "1.0");

void saveGraph(const GraphUtils::GraphData &graphData, const std::string &outputFile);
int main(int argc, char **argv) {
	addArguments(argc, argv);

	const int readEntries = ARGS.get<int>("--n-events");
	const int startEntry = ARGS.get<int>("--start-event");
	const auto inputFiles = ARGS.get<std::vector<std::string>>("--input-files");
	const std::string outputDir = ARGS.get<std::string>("--output-dir");
	const std::string geoConfig = ARGS.get<std::string>("--geo-config");
	const std::string treeName = ARGS.get<std::string>("--tree-name");
	const bool debug = ARGS.get<bool>("--debug");

	auto chain = new TChain(treeName.c_str());
	for (const auto &filename : inputFiles) {
		chain->Add(filename.c_str());
	}
	auto entries = chain->GetEntries();

	NPS::Geometry geometry(geoConfig);

	NPS::simBranches buffer;
	NPS::setBranchAddresses(chain, buffer);

	int processedEntries = 0;
	int savedEvents = 0;
	int useEntries = readEntries < 0 ? chain->GetEntries() : std::min(readEntries, (int)chain->GetEntries());

	const int overlaps = 5;		  // always overlap 5 events
	const int nfeat_perPulse = 2; // energy and time
	const int nclus_perEvent = 2; // only 2 photons in each geant4 event, so at most 2 clusters
	int maxPulses = overlaps * nfeat_perPulse * nclus_perEvent;
	GraphUtils::GraphBuilder graphBuilder(maxPulses, 0, 0, false, true);

	while (processedEntries < useEntries) {

		std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::endl;
		// std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::flush;

		int currEvent = startEntry + processedEntries;
		chain->GetEntry(currEvent);

		auto clust_x = buffer.clust_X->data();
		auto clust_y = buffer.clust_Y->data();
		auto clust_E = buffer.clust_E->data();
		auto clust_Size = buffer.clust_Size->data();

		std::vector<NPS::Cluster> clusters;
		NPS::readSimSignal(*(buffer.clust_Signals), clusters);

		std::map<int, std::vector<double>> blockPulseMap;

		for (const auto &clust : clusters) {
			for (const auto &sig : clust.signals) {

				int blockID_ = sig.blockID; // blockID in Geant4 is different
				int col = blockID_ / NPS::NROWS;
				int row = blockID_ % NPS::NROWS;
				int blockID = geometry.getBlockFromColRow(col, row); // convert to blockID in hcana

				for (const auto &pulse : sig.pulses) {
					blockPulseMap[blockID].push_back(pulse.energy);
					blockPulseMap[blockID].push_back(pulse.time);
				}
			}
		}

		processedEntries++;

		if ((processedEntries) % overlaps == 0) {

			for (auto &[blockID, pulseInfo] : blockPulseMap) {
				int arrSize = pulseInfo.size();
				assert(maxPulses >= pulseInfo.size());
				pulseInfo.resize(maxPulses, 0.0); // padded 0
				graphBuilder.addNode(blockID, pulseInfo);
			}

			auto graphData = graphBuilder.buildGraph();
			saveGraph(graphData, Form("%s/%08d.pt", outputDir.c_str(), savedEvents));
			savedEvents++;
			graphBuilder.reset();
		};
	}

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

void addArguments(int argc, char **argv) {

	ARGS.add_argument("-i", "--input-files")
		.nargs(argparse::nargs_pattern::at_least_one)
		.help("input root files")
		.default_value(std::vector<std::string>{"input.root"})
		.required();

	ARGS.add_argument("-o", "--output-dir").help("output directory").default_value(std::string("./output")).required();

	ARGS.add_argument("-t", "--tree-name")
		.help("name of the tree in the input file")
		.default_value(std::string("nerd"))
		.required();

	ARGS.add_argument("-n", "--n-events")
		.help("number of events to process, -1 for all")
		.default_value(-1)
		.scan<'i', int>();

	ARGS.add_argument("--start-event").help("starting event number").default_value(0).scan<'i', int>();

	ARGS.add_argument("--geo-config")
		.default_value(std::string("database/channel_map.csv"))
		.help("NPS Geometry config file")
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