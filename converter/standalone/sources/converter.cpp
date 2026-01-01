#include "Graph.hh"
#include "NPS.hh"
#include "TChain.h"
#include "TString.h"
#include "Utilities.hh"

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

void addArguments(int argc, char **argv);
argparse::ArgumentParser ARGS("NPS_DataConvertor", "1.0");

void saveGraph(const GraphUtils::GraphData &graphData, const std::string &output_file);
bool isCorruptSignals(const std::vector<std::vector<double>> &signals);
void build_target_edges(GraphUtils::GraphBuilder &builder, std::unordered_map<int, std::vector<int>> &clusterIds);
void buildClusterIdMap(
	int nBlocks, const double *clusterIdArray, std::unordered_map<int, std::vector<int>> &clusterIdsMap
);
template <typename MapLike> std::size_t count_elements(const MapLike &container_map);

int main(int argc, char **argv) {
	addArguments(argc, argv);

	const int readEntries = ARGS.get<int>("--n-events");
	const int startEntry = ARGS.get<int>("--start-event");
	const auto inputFiles = ARGS.get<std::vector<std::string>>("--input-files");
	const std::string outputDir = ARGS.get<std::string>("--output-dir");
	const std::string treeName = ARGS.get<std::string>("--tree-name");
	const int clusMin = ARGS.get<int>("--clus-min");
	const int clusMax = ARGS.get<int>("--clus-max");
	const int sigMin = ARGS.get<int>("--sig-min");
	const int sigMax = ARGS.get<int>("--sig-max");
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
	int useEntries = readEntries < 0 ? chain->GetEntries() : std::min(readEntries, (int)chain->GetEntries());

	GraphUtils::GraphBuilder graphBuilder(NPS::NTIME, 0, 0, false, true);

	auto finishEvent = [&]() {
		graphBuilder.reset();
		processedEntries++;
	};

	while (processedEntries < useEntries) {
		// std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::flush;
		std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::endl;

		int currEvent = startEntry + processedEntries;
		chain->GetEntry(currEvent);

		std::vector<std::vector<double>> signals;
		std::vector<int> blocks;

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
		std::unordered_map<int, std::vector<int>> clusterIds;
		buildClusterIdMap(buffer.Ndata_NPS_cal_fly_block_clusterID, &buffer.NPS_cal_fly_block_clusterID[0], clusterIds);
		build_target_edges(graphBuilder, clusterIds);

		// Apply event selection based on number of clusters and signals
		int nClust = clusterIds.size();
		int nActives = count_elements(clusterIds);
		bool skip = false;
		skip |= nClust <= clusMin || nClust >= clusMax;
		skip |= nActives <= sigMin || nActives >= sigMax;
		if (skip) {
			if (debug) {
				std::cerr << Form(
								 "Skipping event %d with %d clusters and %d active blocks", currEvent, nClust, nActives
							 )
						  << std::endl;
			}
			finishEvent();
			continue;
		}

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

void build_target_edges(GraphUtils::GraphBuilder &builder, std::unordered_map<int, std::vector<int>> &clusterIds) {
	for (const auto &[cid, blockList] : clusterIds) {
		// self-loop for single-block clusters
		if (blockList.size() == 1) {
			builder.addEdgeTarget(blockList[0], blockList[0]);
		} else {
			// fully connect the blocks in the same cluster
			for (size_t i = 0; i < blockList.size(); i++) {
				for (size_t j = i + 1; j < blockList.size(); j++) {
					builder.addEdgeTarget(blockList[i], blockList[j]);
				}
			}
		}
	}
}

void buildClusterIdMap(
	int nBlocks, const double *clusterIdArray, std::unordered_map<int, std::vector<int>> &clusterIdsMap
) {
	for (int block = 0; block < nBlocks; block++) {
		int cid = static_cast<int>(clusterIdArray[block]);
		if (cid == -1) {
			continue;
		}
		clusterIdsMap[cid].push_back(block);
	}
}

template <typename MapLike> std::size_t count_elements(const MapLike &container_map) {
	using MappedType = std::decay_t<decltype(container_map.begin()->second)>;

	static_assert(std::is_member_function_pointer_v<decltype(&MappedType::size)>, "Mapped type must provide size()");

	std::size_t total = 0;
	for (const auto &kv : container_map) {
		total += kv.second.size();
	}
	return total;
}

void addArguments(int argc, char **argv) {

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

	ARGS.add_argument("--clus-min")
		.help("minimum number of clusters to consider event")
		.default_value(0)
		.scan<'i', int>();

	ARGS.add_argument("--clus-max")
		.help("maximum number of clusters to consider event")
		.default_value(1000)
		.scan<'i', int>();

	ARGS.add_argument("--sig-min")
		.help("minimum number of signals to consider event")
		.default_value(0)
		.scan<'i', int>();

	ARGS.add_argument("--sig-max")
		.help("maximum number of signals to consider event")
		.default_value(1000)
		.scan<'i', int>();

	ARGS.add_argument("-d", "--debug").help("debug mode").flag();
	try {
		ARGS.parse_args(argc, argv);
	} catch (const std::runtime_error &err) {
		std::cout << err.what() << std::endl;
		std::cout << ARGS;
		exit(0);
	}
}