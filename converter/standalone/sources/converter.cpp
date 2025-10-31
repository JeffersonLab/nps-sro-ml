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

void add_arguments(int argc, char **argv);
argparse::ArgumentParser ARGS("NPS_DataConvertor", "1.0");

template <typename T, std::size_t N>
void readSignal(
	const int &NSampWaveForm, const std::array<T, N> &SampWaveForm, std::vector<int> &pres,
	std::vector<std::vector<double>> &signals
);

int main(int argc, char **argv) {
	add_arguments(argc, argv);

	const int readEntries = ARGS.get<int>("--n-events");
	const int startEntry = ARGS.get<int>("--start-event");
	const std::string input_dir = ARGS.get<std::string>("--input-dir");
	const std::string output_dir = ARGS.get<std::string>("--output-dir");
	const std::string tree_name = ARGS.get<std::string>("--tree-name");
	const int run = ARGS.get<int>("--run");
	const int seg = ARGS.get<int>("--segment");
	const bool debug = ARGS.get<bool>("--debug");

	std::string filename = Form("%s/nps_hms_coin_%d_%d_1_-1.root", input_dir.c_str(), run, seg);

	auto chain = new TChain(tree_name.c_str());
	chain->Add(filename.c_str());
	auto entries = chain->GetEntries();

	npsBranches buffer;
	setBranchAddresses(chain, buffer);

	int processedEntries = 0;
	int useEntries = readEntries == -1 ? chain->GetEntries() : std::min(readEntries, (int)chain->GetEntries());

	GraphBuilder graphBuilder(NTIME);

	while (processedEntries < useEntries) {
		// std::cout << "Processing entry " << processedEntries << "/" << useEntries << "\n";
		std::cout << "\rProcessing entry " << processedEntries + 1 << "/" << useEntries << std::flush;
		chain->GetEntry(startEntry + processedEntries);

		if (buffer.Ndata_NPS_cal_fly_adcSampWaveform > NDATA) {
			std::cerr << "Warning: Ndata_NPS_cal_fly_adcSampWaveform > NDATA ("
					  << buffer.Ndata_NPS_cal_fly_adcSampWaveform << " > " << NDATA << "). Skipping event "
					  << processedEntries << ".\n";
			processedEntries++;
			continue;
		}

		std::vector<std::vector<double>> signals;
		std::vector<int> blocks;

		readSignal(buffer.Ndata_NPS_cal_fly_adcSampWaveform, buffer.NPS_cal_fly_adcSampWaveform, blocks, signals);

		if (signals.size() == 0) {
			std::cerr << "Warning: no signals found. Skipping event " << processedEntries << ".\n";
			processedEntries++;
			continue;
		}
		if (signals.size() > 0 && signals[0].size() != NTIME) {
			std::cerr << "Warning: signals size is " << signals[0].size() << " instead of 110. Skipping event "
					  << processedEntries << ".\n";
			processedEntries++;
			continue;
		}
		graphBuilder.addNodes(blocks, signals);

		// add target edges
		int expectedEdges = 0;
		std::map<int, std::vector<int>> clusterIds;
		for (int block = 0; block < buffer.NPS_cal_fly_block_clusterID.size(); block++) {
			int cid = static_cast<int>(buffer.NPS_cal_fly_block_clusterID[block]);
			if (cid == -1) {
				continue;
			}
			clusterIds[cid].push_back(block);
		}
		for (const auto &[cid, blockList] : clusterIds) {
			expectedEdges += blockList.size() * (blockList.size() - 1) / 2;

			for (size_t i = 0; i < blockList.size(); i++) {
				for (size_t j = 0; j < blockList.size(); j++) {
					if (i != j) {
						graphBuilder.addEdgeTarget(blockList[i], blockList[j]);
					}
				}
			}
		}

		auto graphData = graphBuilder.buildGraph();
		auto nodeIds = toTensor(graphData.nodeIds);				// [num_nodes]
		auto nodeFeatures = toTensor2D(graphData.nodeFeatures); // [num_nodes][num_node_features]
		auto nodeTargets = toTensor2D(graphData.nodeTargets);	// [num_nodes][num_targets]
		auto edgeFeatures = toTensor2D(graphData.edgeFeatures); // [num_edges][num_edge_features]
		auto edgeTargetFeatures =
			toTensor2D(graphData.edgeTargetFeatures);				  // [num_target_edges][num_target_edge_features]
		auto edgeIndex = toTensor2D(graphData.edgeIndex);			  // [2][num_edges]
		auto edgeTargetIndex = toTensor2D(graphData.edgeTargetIndex); // [2][num_target_edges]


		std::string output_file = Form("%s/%04d/%d/%08d.pt", output_dir.c_str(), run, seg, processedEntries);
		std::filesystem::create_directories(std::filesystem::path(output_file).parent_path());
		saveTensors(
			output_file, nodeIds, nodeFeatures, nodeTargets, edgeFeatures, edgeTargetFeatures, edgeIndex,
			edgeTargetIndex
		);
		graphBuilder.reset();
		processedEntries++;
	}
	return 0;
}

template <typename T, std::size_t N>
void readSignal(
	const int &NSampWaveForm, const std::array<T, N> &SampWaveForm, std::vector<int> &blocks,
	std::vector<std::vector<double>> &signals
) {

	signals.clear();
	blocks.clear();
	std::map<int, bool> block_seen;

	int ns = 0;

	while (ns < NSampWaveForm) {
		int bloc = SampWaveForm[ns++];	// bloc number (actually the slot number)
		int nsamp = SampWaveForm[ns++]; // number of time samples for this bloc = NTIME = 110

		// check for correct unpacking of the waveform
		if (ns + nsamp > NSampWaveForm) {
			std::cerr << "Warning: not enough samples for block " << bloc << " (expected " << nsamp << ", available "
					  << (NSampWaveForm - ns) << "). Stopping readSignal.\n";
			break;
		}

		// keep for historical reasons
		if (bloc == 2000) {
			bloc = 1080;
		}
		if (bloc == 2001) {
			bloc = 1081;
		}

		if (bloc >= 0 && bloc < NBLOCKS) {

			if (block_seen.count(bloc) > 0 && block_seen[bloc]) {
				// this should never happen
				ns += nsamp;
				continue;
			}

			block_seen[bloc] = true;
			blocks.push_back(bloc);

			std::vector<double> sig;
			for (int it = 0; it < nsamp; it++) {
				sig.push_back(SampWaveForm[ns++]);
			}
			signals.push_back(sig);
		} else {
			ns += nsamp; // skip invalid bloc
		}
	}
	return;
}

void add_arguments(int argc, char **argv) {

	ARGS.add_argument("-i", "--input-dir").help("input directory").default_value(std::string("./cache")).required();

	ARGS.add_argument("-o", "--output-dir").help("output directory").default_value(std::string("./output")).required();

	ARGS.add_argument("-r", "--run").help("run number").nargs(1).default_value(4454).scan<'i', int>().required();

	ARGS.add_argument("-s", "--segment").help("segment number").nargs(1).default_value(0).scan<'i', int>().required();

	ARGS.add_argument("-t", "--tree-name")
		.help("name of the tree in the input file")
		.default_value(std::string("T"))
		.required();

	ARGS.add_argument("-n", "--n-events")
		.help("number of events to process, -1 for all")
		.default_value(-1)
		.scan<'i', int>();

	ARGS.add_argument("--start-event").help("starting event number").default_value(0).scan<'i', int>();

	ARGS.add_argument("-d", "--debug").help("debug mode").flag();
	try {
		ARGS.parse_args(argc, argv);
	} catch (const std::runtime_error &err) {
		std::cout << err.what() << std::endl;
		std::cout << ARGS;
		exit(0);
	}
}