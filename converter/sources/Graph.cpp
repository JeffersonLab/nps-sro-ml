#include "Graph.hh"

namespace GraphUtils {

std::unordered_map<int, std::vector<int>> buildAdjacencyList(const std::vector<std::vector<int>> &edgeIndex) {

	std::unordered_map<int, std::vector<int>> adjList;
	for (size_t i = 0; i < edgeIndex[0].size(); ++i) {
		auto src = edgeIndex[0][i];
		auto dst = edgeIndex[1][i];
		adjList[src].push_back(dst);
		// no need for dst->src, assume edgeIndex already has both directions if undirected
	}
	return adjList;
}

int countNodes(const std::vector<std::vector<int>> &edgeIndex) {
	std::unordered_set<int> nodeSet;
	for (size_t i = 0; i < edgeIndex[0].size(); ++i) {
		nodeSet.insert(edgeIndex[0][i]);
		nodeSet.insert(edgeIndex[1][i]);
	}
	return nodeSet.size();
}

int getLargestNodeId(const std::vector<std::vector<int>> &edgeIndex) {
	int maxId = -1;
	for (size_t i = 0; i < edgeIndex[0].size(); ++i) {
		maxId = std::max(maxId, edgeIndex[0][i]);
		maxId = std::max(maxId, edgeIndex[1][i]);
	}
	return maxId;
}

std::vector<std::vector<int>> getConnectedComponents(const std::vector<std::vector<int>> &edgeIndex) {

	std::vector<std::vector<int>> components;
	// allocate "visited" based on largest node id in edgeIndex
	int graphSize = getLargestNodeId(edgeIndex) + 1;
	std::vector<bool> visited(graphSize, false);
	std::unordered_map<int, std::vector<int>> adjList = buildAdjacencyList(edgeIndex);

	std::function<void(int, std::vector<int> &)> dfs = [&](int node, std::vector<int> &component) -> void {
		visited[node] = true;
		component.push_back(node);

		if (adjList.find(node) == adjList.end()) {
			return;
		}

		for (int neighbor : adjList[node]) {
			if (!visited[neighbor]) {
				dfs(neighbor, component);
			}
		}
	};

	// iterate through nodes in adjList instead of all numNodes
	// this handles isolated nodes with self-loops correctly
	// for (size_t i = 0; i < adjList.size(); i++) {
	for (const auto &[node, _] : adjList) {
		if (!visited[node]) {
			std::vector<int> component;
			dfs(node, component);
			components.push_back(component);
		}
	}
	return components;
}

GraphBuilder::GraphBuilder(
	int num_node_features, int num_targets, int num_edge_attributes, int num_dimensions, bool directed, bool self_loops,
	bool fill_edges
) :
	mNumNodeFeatures(num_node_features),
	mNumTargets(num_targets),
	mNumEdgeAttributes(num_edge_attributes),
	mNumDimensions(num_dimensions),
	mBuilt(false),
	mDirected(directed),
	mIncludeSelfLoops(self_loops),
	mFillEdges(fill_edges) {}

GraphBuilder::~GraphBuilder() {}

void GraphBuilder::addNode(int id, const std::vector<double> &features) {
	if (features.size() != mNumNodeFeatures) {
		throw std::invalid_argument("Feature size does not match number of features");
	}
	if (mNodeFeatures.count(id) > 0) {
		throw std::invalid_argument("Node with the same ID already exists");
	}
	mNodeFeatures[id] = features;
}

void GraphBuilder::addNode(int id, std::initializer_list<double> features) {
	addNode(id, std::vector<double>(features));
}

void GraphBuilder::addNodes(const std::vector<int> &ids, const std::vector<std::vector<double>> &features) {
	if (ids.size() != features.size()) {
		throw std::invalid_argument("IDs size does not match features size");
	}
	for (size_t i = 0; i < ids.size(); ++i) {
		addNode(ids[i], features[i]);
	}
}

void GraphBuilder::addNodeTarget(int id, const std::vector<double> &target) {
	if (mNodeFeatures.count(id) == 0) {
		throw std::invalid_argument("Node does not exist");
	}
	mNodeTargets[id] = target;
}

void GraphBuilder::addNodeTargets(const std::vector<int> &ids, const std::vector<std::vector<double>> &targets) {
	if (ids.size() != targets.size()) {
		throw std::invalid_argument("IDs size does not match targets size");
	}
	for (size_t i = 0; i < ids.size(); ++i) {
		addNodeTarget(ids[i], targets[i]);
	}
}

void GraphBuilder::addNodePosition(int id, const std::vector<double> &position) {
	if (mNodeFeatures.count(id) == 0) {
		throw std::invalid_argument("Node does not exist");
	}
	if (static_cast<int>(position.size()) != mNumDimensions) {
		throw std::invalid_argument("Position size does not match number of dimensions");
	}
	mNodePositions[id] = position;
}

void GraphBuilder::addNodePositions(const std::vector<int> &ids, const std::vector<std::vector<double>> &positions) {
	if (ids.size() != positions.size()) {
		throw std::invalid_argument("IDs size does not match positions size");
	}
	for (size_t i = 0; i < ids.size(); ++i) {
		addNodePosition(ids[i], positions[i]);
	}
}

void GraphBuilder::addEdge(int src_id, int dst_id) {
	if (mNodeFeatures.count(src_id) == 0 || mNodeFeatures.count(dst_id) == 0) {
		throw std::invalid_argument("Source or destination node does not exist");
	}

	// if seen in dst -> src, skip if undirected graph is considered
	bool seen = mEdges.count(dst_id) && mEdges.at(dst_id).count(src_id);
	if (!mDirected && seen) {
		return;
	}
	if (!mIncludeSelfLoops && src_id == dst_id) {
		return;
	}
	mEdges[src_id].insert(dst_id);
}

void GraphBuilder::addEdge(int src_id, int dst_id, const std::vector<double> &attributes) {
	addEdge(src_id, dst_id);
	if (attributes.size() != mNumEdgeAttributes) {
		throw std::invalid_argument("Edge attribute size does not match number of edge attributes");
	}
	if (mEdgeAttributes.count({src_id, dst_id}) >= 1) {
		throw std::invalid_argument("Edge with the same source and destination already has attributes");
	}
	mEdgeAttributes[{src_id, dst_id}] = attributes;
}

void GraphBuilder::addEdges(const std::vector<int> &node_ids, const std::string &algorithm) {
	if (algorithm == "fully_connected") {
		for (size_t i = 0; i < node_ids.size(); i++) {
			for (size_t j = i + 1; j < node_ids.size(); j++) {
				addEdge(node_ids[i], node_ids[j]);
			}
		}
	} else if (algorithm == "center_to_neighbor") {
		// assume the first node is the center node and connect it to all other nodes
		int center_id = node_ids[0];
		for (size_t i = 1; i < node_ids.size(); i++) {
			addEdge(center_id, node_ids[i]);
		}
	} else {
		throw std::invalid_argument("Unsupported edge creation algorithm");
	}
}

void GraphBuilder::reset() {
	mNodeFeatures.clear();
	mNodeTargets.clear();
	mNodePositions.clear();
	mEdges.clear();
	mEdgeAttributes.clear();
	mBuilt = false;
}

bool GraphBuilder::validate() const {

	// ensure mNodeFeatures, mNodeTargets, mNodePositions have consistent ids
	for (const auto &[id, _] : mNodeTargets) {
		if (mNodeFeatures.count(id) == 0) {
			return false;
		}
		if (mNodePositions.count(id) == 0) {
			return false;
		}
	}

	// Only validate edges if fill_edges is true, otherwise edges is not mandatory as mNodeTargets and mNodePositions
	// can be used to construct a graph without edges. Edge index can be optionally filled using different algorithms
	// (e.g. fully connected, denter-to-neighbor, etc.).
	if (mFillEdges) {

		// ensure all edges have valid source and destination nodes
		for (const auto &[src_id, neighbors] : mEdges) {
			if (mNodeFeatures.count(src_id) == 0) {
				return false;
			}
			for (const auto &dst_id : neighbors) {
				if (mNodeFeatures.count(dst_id) == 0) {
					return false;
				}
			}
		}

		// ensure all edge attributes have valid source and destination nodes
		for (const auto &[edge, attributes] : mEdgeAttributes) {
			const auto &[src_id, dst_id] = edge;
			if (mNodeFeatures.count(src_id) == 0 || mNodeFeatures.count(dst_id) == 0) {
				return false;
			}
		}
	}
	return true;
}

GraphUtils::GraphData GraphBuilder::buildGraph() {

	if (validate() == 0) {
		throw std::runtime_error("Graph validation failed");
	}

	GraphUtils::GraphData graph{
		.nodeFeatures = {},
		.edgeIndex = {{}, {}},
		.edgeAttributes = {},
		.nodeTargets = {},
		.nodePositions = {},
	};

	std::unordered_map<int, int> nodeIdToIndex; // mapping from node id to index in the final graph
	int currentIndex = 0;

	for (const auto &[id, features] : mNodeFeatures) {
		graph.nodeFeatures.push_back(features);
		if (mNodeTargets.count(id)) {
			graph.nodeTargets.push_back(mNodeTargets.at(id));
		} else {
			graph.nodeTargets.push_back(std::vector<double>{});
		}
		if (mNodePositions.count(id)) {
			graph.nodePositions.push_back(mNodePositions.at(id));
		} else {
			graph.nodePositions.push_back(std::vector<double>{});
		}
		nodeIdToIndex[id] = currentIndex++;
	}

	// edges is now just coo
	// no duplicates allowed in the set
	// 1 -> [2,3,4,5]
	// 2 -> [1,2,3]
	// ...

	// build edges
	for (const auto &[src_id_, neighbors] : mEdges) {
		for (const auto &dst_id_ : neighbors) {

			// map whatever Id used in main code to local index
			int src_id = nodeIdToIndex[src_id_];
			int dst_id = nodeIdToIndex[dst_id_];
			graph.edgeIndex[0].push_back(src_id);
			graph.edgeIndex[1].push_back(dst_id);

			// add edge atttributes if exist here so the order matches edgeIndex
			if (mEdgeAttributes.count({src_id_, dst_id_}) ||
				(!mDirected && mEdgeAttributes.count({dst_id_, src_id_}))) {
				graph.edgeAttributes.push_back(
					mEdgeAttributes.count({src_id_, dst_id_}) ? mEdgeAttributes.at({src_id_, dst_id_})
															  : mEdgeAttributes.at({dst_id_, src_id_})
				);
			} else {
				graph.edgeAttributes.push_back(std::vector<double>{});
			}
		}
	}

	mBuilt = true;
	return std::move(graph);
}

int GraphBuilder::getNumEdges() const {
	int count = 0;
	for (const auto &[src_id, neighbors] : mEdges) {
		count += neighbors.size();
	}
	return count;
}

void GraphBuilder::PrintGraph() const {
	for (const auto &[id, features] : mNodeFeatures) {
		std::cout << "Node " << id << ": Features = [";
		for (const auto &f : features) {
			std::cout << f << " ";
		}
		std::cout << "]";
		if (mNodeTargets.count(id)) {
			std::cout << ", Target = [";
			for (const auto &t : mNodeTargets.at(id)) {
				std::cout << t << " ";
			}
			std::cout << "]";
		}
		if (mNodePositions.count(id)) {
			std::cout << ", Position = [";
			for (const auto &p : mNodePositions.at(id)) {
				std::cout << p << " ";
			}
			std::cout << "]";
		}
		std::cout << std::endl;
	}

	if (mFillEdges) {
		std::cout << "Edges:" << std::endl;
		for (const auto &[src_id, neighbors] : mEdges) {
			for (const auto &dst_id : neighbors) {
				std::cout << src_id << " -> " << dst_id;
				if (mEdgeAttributes.count({src_id, dst_id})) {
					std::cout << " [Attributes = ";
					for (const auto &attr : mEdgeAttributes.at({src_id, dst_id})) {
						std::cout << attr << " ";
					}
					std::cout << "]";
				}
				std::cout << std::endl;
			}
		}
	}
}

}; // namespace GraphUtils