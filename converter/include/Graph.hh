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
	// data structure analogous to PyG's Data object, see
	// https://pytorch-geometric.readthedocs.io/en/2.6.1/get_started/introduction.html#data-handling-of-graphs.

	std::vector<std::vector<double>> nodeFeatures;	 // [num_nodes][num_node_features]
	std::vector<std::vector<int>> edgeIndex;		 // [2][num_edges]
	std::vector<std::vector<double>> edgeAttributes; // [num_edges][num_edge_features]
	std::vector<std::vector<double>> nodeTargets;	 // [num_nodes][num_targets]
	std::vector<std::vector<double>> nodePositions;	 //[num_nodes][num_dimensions]
};

int countNodes(const std::vector<std::vector<int>> &edgeIndex);
int getLargestNodeId(const std::vector<std::vector<int>> &edgeIndex);
std::unordered_map<int, std::vector<int>> buildAdjacencyList(const std::vector<std::vector<int>> &edgeIndex);
std::vector<std::vector<int>> getConnectedComponents(const std::vector<std::vector<int>> &edgeIndex);

class GraphBuilder {
public:
	GraphBuilder(
		int num_node_features, int num_targets = 1, int num_edge_attributes = 0, int num_dimensions = 2,
		bool directed = false, bool self_loops = false, bool fill_edges = false
	);
	~GraphBuilder();

	void addNodes(const std::vector<int> &ids, const std::vector<std::vector<double>> &features);
	void addNode(int id, const std::vector<double> &features);
	void addNode(int id, std::initializer_list<double> features);

	void addNodeTargets(const std::vector<int> &ids, const std::vector<std::vector<double>> &targets);
	void addNodeTarget(int id, const std::vector<double> &target);

	void addNodePositions(const std::vector<int> &ids, const std::vector<std::vector<double>> &positions);
	void addNodePosition(int id, const std::vector<double> &position);

	void addEdge(int src_id, int dst_id);
	void addEdge(int src_id, int dst_id, const std::vector<double> &attributes);
	void addEdges(const std::vector<int> &node_ids, const std::string &algorithm = "fully_connected");

	GraphUtils::GraphData buildGraph();
	void reset();
	bool validate() const;

	void PrintGraph() const;

	inline bool isDirected() const noexcept;
	inline bool hasSelfLoops() const noexcept;
	inline bool isBuilt() const noexcept;

	int getNumNodeFeatures() const { return mNumNodeFeatures; }
	int getNumEdgeAttributes() const { return mNumEdgeAttributes; }
	int getNumTargets() const { return mNumTargets; }
	int getNumDimensions() const { return mNumDimensions; }

	int getNumNodes() const { return mNodeFeatures.size(); }
	int getNumEdges() const;

private:
	int mNumNodeFeatures;
	int mNumTargets;
	int mNumEdgeAttributes;
	int mNumDimensions;

	bool mBuilt;
	bool mDirected;
	bool mIncludeSelfLoops;
	bool mFillEdges;

	std::map<int, std::vector<double>> mNodeFeatures;  // node_id -> features
	std::map<int, std::vector<double>> mNodeTargets;   // node_id -> target
	std::map<int, std::vector<double>> mNodePositions; // node_id -> position

	std::map<int, std::unordered_set<int>> mEdges;						// src_id -> set of dst_ids
	std::map<std::pair<int, int>, std::vector<double>> mEdgeAttributes; // (src_id, dst_id) -> edge attributes
};

inline bool GraphBuilder::isDirected() const noexcept { return mDirected; }
inline bool GraphBuilder::hasSelfLoops() const noexcept { return mIncludeSelfLoops; }
inline bool GraphBuilder::isBuilt() const noexcept { return mBuilt; }
}; // namespace GraphUtils

#endif // GRAPH_HH