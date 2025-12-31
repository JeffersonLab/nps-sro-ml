#ifndef GRAPH_HH
#define GRAPH_HH

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ios>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace GraphUtils {

struct GraphData {
	std::vector<int> nodeIds;
	std::vector<std::vector<double>> nodeFeatures;		 // [num_nodes][num_node_features]
	std::vector<std::vector<double>> nodeTargets;		 // [num_nodes][num_targets]
	std::vector<std::vector<double>> edgeFeatures;		 // [num_edges][num_edge_features]
	std::vector<std::vector<double>> edgeTargetFeatures; // [num_target_edges][num_edge_target_features]
	std::vector<std::vector<int>> edgeIndex;			 // [2][num_edges]
	std::vector<std::vector<int>> edgeTargetIndex;		 // [2][num_target_edges]
};

int countNodes(const std::vector<std::vector<int>> &edgeIndex);
int getLargestNodeId(const std::vector<std::vector<int>> &edgeIndex);
std::unordered_map<int, std::vector<int>> buildAdjacencyList(const std::vector<std::vector<int>> &edgeIndex);
std::vector<std::vector<int>> getConnectedComponents(const std::vector<std::vector<int>> &edgeIndex);

class GraphBuilder {
public:
	GraphBuilder(
		int n_features = 110, int n_edge_features = 0, int n_target_edge_features = 0, bool directed = false,
		bool self_loops = false
	);
	~GraphBuilder();

	void addNodes(std::vector<int> ids, std::vector<std::vector<double>> features);
	void addNode(int id, const std::vector<double> &features);
	void addNode(int id, std::initializer_list<double> features);
	void addNodeTargets(std::vector<int> ids, std::vector<std::vector<double>> targets);
	void addNodeTarget(int id, const std::vector<double> &target);

	void addEdge(int src_id, int dst_id);
	void addEdge(int src_id, int dst_id, const std::vector<double> &features);
	void addEdgeTarget(int src_id, int dst_id);
	void addEdgeTarget(int src_id, int dst_id, const std::vector<double> &target);

	GraphUtils::GraphData buildGraph();
	void reset();

	inline bool isDirected() const noexcept;
	inline bool hasSelfLoops() const noexcept;
	inline bool isBuilt() const noexcept;

	void summary() const {
		std::cout << "Graph Summary:\n"
				  << "  Nodes: " << mNodeFeatures.size() << "\n"
				  << "  Edges: " << mEdges.size() << "\n"
				  << "  Edges Target: " << mEdgeTargets.size() << "\n"
				  << "  Directed: " << std::boolalpha << mDirected << "\n"
				  << "  Self-loops: " << mIncludeSelfLoops << "\n";
	}

	int getNumNodeFeatures() const { return mNumNodeFeatures; }
	int getNumEdgeFeatures() const { return mNumEdgeFeatures; }
	int getNumNodes() const { return mNodeFeatures.size(); }
	int getNumEdges() const;
	int getNumEdgeTargets() const;

private:
	int mNumNodeFeatures;
	int mNumEdgeFeatures;
	int mNumTargetEdgeFeatures;
	bool mBuilt;
	bool mDirected;
	bool mIncludeSelfLoops;

	std::map<int, std::vector<double>> mNodeFeatures; // node_id -> features
	std::map<int, std::vector<double>> mNodeTargets;  // node_id -> target values

	std::map<int, std::unordered_set<int>> mEdges;							// src_id -> set of dst_ids
	std::map<int, std::unordered_set<int>> mEdgeTargets;					// src_id -> set of dst_ids
	std::map<std::pair<int, int>, std::vector<double>> mEdgeFeatures;		// (src_id, dst_id) -> features
	std::map<std::pair<int, int>, std::vector<double>> mEdgeTargetFeatures; // (src_id, dst_id) -> target features
};

inline bool GraphBuilder::isDirected() const noexcept { return mDirected; }
inline bool GraphBuilder::hasSelfLoops() const noexcept { return mIncludeSelfLoops; }
inline bool GraphBuilder::isBuilt() const noexcept { return mBuilt; }
}; // namespace GraphUtils

#endif // GRAPH_HH