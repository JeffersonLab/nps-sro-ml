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
	const std::string geoConfig = ARGS.get<std::string>("--geo-config");
	const bool createEdges = ARGS.get<bool>("--edge-creation");
	const std::string edgeAlgorithm = ARGS.get<std::string>("--edge-algorithm");
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

	NPS::Geometry geometry(geoConfig);

	NPS::npsBranches buffer;
	NPS::setBranchAddresses(chain, buffer);

	int processedEntries = 0;
	int savedEvents = 0;
	int useEntries = readEntries < 0 ? chain->GetEntries() : std::min(readEntries, (int)chain->GetEntries());
	auto isDirected = edgeAlgorithm == "center_to_neighbor";

	GraphUtils::GraphBuilder graphBuilder(NPS::NTIME, 1, 0, 2, isDirected, true, createEdges);

	auto finishEvent = [&]() {
		graphBuilder.reset();
		processedEntries++;
	};

	while (processedEntries < useEntries) {
		std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::flush;

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

		for (const auto &block : blocks) {
			auto [col, row] = geometry.getColRowFromBlock(block);
			graphBuilder.addNodePosition(block, {static_cast<double>(col), static_cast<double>(row)});
		}

		std::unordered_map<int, std::vector<int>> clusterIds;
		buildClusterIdMap(buffer.Ndata_NPS_cal_fly_block_clusterID, &buffer.NPS_cal_fly_block_clusterID[0], clusterIds);

		// There is no overlapping clusters in this dataset, so each cluster corresponds to one connected component. We
		// can directly use cluster ID as node target without building connected components.
		for (const auto &[cid, nodes] : clusterIds) {
			for (int node_id : nodes) {
				graphBuilder.addNodeTarget(node_id, {static_cast<double>(cid)});
			}
		}

		if (createEdges) {

			for (const auto &[cid, nodes] : clusterIds) {

                if (nodes.size() == 1) {
                    graphBuilder.addEdge(nodes[0], nodes[0]);
                }
                else {
				    graphBuilder.addEdges(nodes, edgeAlgorithm);
                }
                
			}
		}

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

        if (graphBuilder.isEmpty()) {
    		finishEvent();
            continue;
        }

		if (debug && createEdges) {
			// this is only valid if there is no overlapping clusters.
			auto components = GraphUtils::getConnectedComponents(graphData.edgeIndex);
			assert(components.size() == nClust);
		}
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
	auto nodeFeatures = TorchUtils::toTensor2D(graphData.nodeFeatures);		// [num_nodes][num_node_features]
	auto nodeTargets = TorchUtils::toTensor2D(graphData.nodeTargets);		// [num_nodes][num_targets]
	auto edgeAttributes = TorchUtils::toTensor2D(graphData.edgeAttributes); // [num_edges][num_edge_features]
	auto edgeIndex = TorchUtils::toTensor2D(graphData.edgeIndex);			// [2][num_edges]
	auto nodePositions = TorchUtils::toTensor2D(graphData.nodePositions);	// [num_nodes][num_dimensions]

	std::filesystem::create_directories(std::filesystem::path(outputFile).parent_path());
	TorchUtils::saveTensors(outputFile, nodeFeatures, edgeIndex, edgeAttributes, nodeTargets, nodePositions);
}

bool isCorruptSignals(const std::vector<std::vector<double>> &signals) {
	bool corrupt = false;
	corrupt |= signals.size() == 0;
	corrupt |= signals.size() > 0 && signals[0].size() != NPS::NTIME;
	return corrupt;
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

	ARGS.add_argument("--edge-creation").help("whether to create edges or not").flag();

	ARGS.add_argument("--edge-algorithm")
		.help("algorithm to create edges (fully_connected, center_to_neighbor, etc.)")
        .choices("fully_connected", "center_to_neighbor")
		.default_value(std::string("fully_connected"))
		.required();

	ARGS.add_argument("--start-event").help("starting event number").default_value(0).scan<'i', int>();

	ARGS.add_argument("--geo-config")
		.default_value(std::string("database/channel_map.csv"))
		.help("NPS Geometry config file")
		.required();
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