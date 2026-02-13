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
void build_target_edges(GraphUtils::GraphBuilder &builder, std::unordered_map<int, std::vector<int>> &clusterIds);
int main(int argc, char **argv) {
	addArguments(argc, argv);

	const int readEntries = ARGS.get<int>("--n-events");
	const int startEntry = ARGS.get<int>("--start-event");
	const auto inputFiles = ARGS.get<std::vector<std::string>>("--input-files");
	const int overlaps = ARGS.get<int>("--overlaps");
	const double timeGap = ARGS.get<double>("--dt");
	const std::string outputDir = ARGS.get<std::string>("--output-dir");
	const std::string geoConfig = ARGS.get<std::string>("--geo-config");
	const std::string treeName = ARGS.get<std::string>("--tree-name");
	const bool createEdges = ARGS.get<bool>("--edge-creation");
	const std::string edgeAlgorithm = ARGS.get<std::string>("--edge-algorithm");
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

	const int nfeat_perPulse = 2; // energy and time
	const int nclus_perEvent = 2; // only 2 photons in each geant4 event, so at most 2 clusters
	int maxPulses = overlaps * nfeat_perPulse * nclus_perEvent;
	GraphUtils::GraphBuilder graphBuilder(maxPulses, 1, 0, 2, false, true, createEdges);

	// buffer for overlapping events
	std::vector<NPS::Cluster> clusters;

	while (processedEntries < useEntries) {

		std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::endl;
		// std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::flush;

		int currEvent = startEntry + processedEntries;
		chain->GetEntry(currEvent);

		auto clust_x = buffer.clust_X->data();
		auto clust_y = buffer.clust_Y->data();
		auto clust_E = buffer.clust_E->data();
		auto clust_Size = buffer.clust_Size->data();

		std::vector<NPS::Cluster> clusters_buffer;
		NPS::readSimSignal(*(buffer.clust_Signals), clusters_buffer);
		clusters.insert(clusters.end(), clusters_buffer.begin(), clusters_buffer.end());

		int eventIndex = processedEntries % overlaps;

		processedEntries++;

		if (processedEntries % overlaps == 0) {

			// build graph for the current set of overlapping events
			std::unordered_map<int, std::vector<int>> clustToNodeIds;

			std::map<int, std::vector<double>>
				blockPulseMap; // blockID to [energy, time, energy, time, ...] for all pulses

			std::unordered_map<int, int> nodeToBlockId;
			std::unordered_map<int, int> nodeToClusterId;

			int nodeId = 0;
			for (size_t i = 0; i < clusters.size(); i++) {
				const auto &cluster = clusters[i];

				for (const auto &signal : cluster.signals) {
					int blockId_ = signal.blockID;
					auto [col, row] = geometry.getColRowFromBlock(blockId_);
					int blockId = geometry.getBlockFromColRow(col, row);

					clustToNodeIds[i + 1].push_back(nodeId);
					nodeToBlockId[nodeId] = blockId;
					nodeToClusterId[nodeId] = i + 1;
					nodeId++;

					for (const auto &pulse : signal.pulses) {
						blockPulseMap[blockId].push_back(pulse.energy);
						blockPulseMap[blockId].push_back(pulse.time);
					}
				}
			}

			for (const auto &[nodeId, blockId] : nodeToBlockId) {
				assert(maxPulses >= blockPulseMap[blockId].size()); // ensure we have enough features to hold all pulses
				blockPulseMap[blockId].resize(maxPulses, 0.0);
				graphBuilder.addNode(nodeId, blockPulseMap[blockId]);
				auto [col, row] = geometry.getColRowFromBlock(blockId);
				graphBuilder.addNodePosition(nodeId, {static_cast<double>(col), static_cast<double>(row)});
				graphBuilder.addNodeTarget(nodeId, {static_cast<double>(nodeToClusterId[nodeId])});
			}

			if (createEdges) {
				for (const auto &[cid, nodes] : clustToNodeIds) {
					graphBuilder.addEdges(nodes, edgeAlgorithm);
				}
			}

			auto graphData = graphBuilder.buildGraph();
			saveGraph(graphData, Form("%s/%08d.pt", outputDir.c_str(), savedEvents));
			if (debug && savedEvents == 0) {
				graphBuilder.PrintGraph();
			}
			savedEvents++;
			graphBuilder.reset();
			clusters.clear();
		};
	}

	return 0;
}

void saveGraph(const GraphUtils::GraphData &graphData, const std::string &outputFile) {
	auto nodeFeatures = TorchUtils::toTensor2D(graphData.nodeFeatures);		// [num_nodes][num_node_features]
	auto nodeTargets = TorchUtils::toTensor2D(graphData.nodeTargets);		// [num_nodes][num_targets]
	auto edgeAttributes = TorchUtils::toTensor2D(graphData.edgeAttributes); // [num_edges][num_edge_features]
	auto edgeIndex = TorchUtils::toTensor2D(graphData.edgeIndex);			// [2][num_edges]
	auto nodePositions = TorchUtils::toTensor2D(graphData.nodePositions);	// [num_nodes][num_dimensions]

	std::filesystem::create_directories(std::filesystem::path(outputFile).parent_path());
	TorchUtils::saveTensors(outputFile, nodeFeatures, edgeIndex, edgeAttributes, nodeTargets, nodePositions);
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

	ARGS.add_argument("--overlaps")
		.help("number of overlapping events to mix together")
		.default_value(5)
		.scan<'i', int>();

	ARGS.add_argument("--dt").help("time gap between overlapping events in ns").default_value(32.0).scan<'g', double>();

	ARGS.add_argument("--edge-creation").help("whether to create edges or not").flag();

	ARGS.add_argument("--edge-algorithm")
		.help("algorithm to create edges (fully_connected, center_to_neighbor, etc.)")
		.default_value(std::string("fully_connected"))
		.required();

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