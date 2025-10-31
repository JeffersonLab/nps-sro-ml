#include "Graph.hh"

GraphBuilder::GraphBuilder(
	int n_features, int n_edge_features, int n_target_edge_features, bool directed, bool self_loops
) :
	mNumNodeFeatures(n_features),
	mNumEdgeFeatures(n_edge_features),
	mNumTargetEdgeFeatures(n_target_edge_features),
	mBuilt(false),
	mDirected(directed),
	mIncludeSelfLoops(self_loops) {}

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

void GraphBuilder::addNodes(std::vector<int> ids, std::vector<std::vector<double>> features) {
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

void GraphBuilder::addNodeTargets(std::vector<int> ids, std::vector<std::vector<double>> targets) {
	if (ids.size() != targets.size()) {
		throw std::invalid_argument("IDs size does not match targets size");
	}
	for (size_t i = 0; i < ids.size(); ++i) {
		addNodeTarget(ids[i], targets[i]);
	}
}

void GraphBuilder::addEdge(int src_id, int dst_id) {
	if (mNodeFeatures.count(src_id) == 0 || mNodeFeatures.count(dst_id) == 0) {
		throw std::invalid_argument("Source or destination node does not exist");
	}

	bool seen = mEdges.count(dst_id) && mEdges.at(dst_id).count(src_id);
	if (!mDirected && seen) {
		return;
	}
	if (!mIncludeSelfLoops && src_id == dst_id) {
		return;
	}
	mEdges[src_id].insert(dst_id);
}

void GraphBuilder::addEdge(int src_id, int dst_id, const std::vector<double> &features) {
	addEdge(src_id, dst_id);
	if (features.size() != mNumEdgeFeatures) {
		throw std::invalid_argument("Edge feature size does not match number of edge features");
	}
	if (mEdgeFeatures.count({src_id, dst_id}) >= 1) {
		throw std::invalid_argument("Edge with the same source and destination already has features");
	}
	mEdgeFeatures[{src_id, dst_id}] = features;
}

void GraphBuilder::addEdgeTarget(int src_id, int dst_id) {
	if (mNodeFeatures.count(src_id) == 0 || mNodeFeatures.count(dst_id) == 0) {
		throw std::invalid_argument("Source or destination node does not exist");
	}
	bool seen = mEdgeTargets.count(dst_id) && mEdgeTargets.at(dst_id).count(src_id);
	if (!mDirected && seen) {
		return;
	}
	if (!mIncludeSelfLoops && src_id == dst_id) {
		return;
	}
	mEdgeTargets[src_id].insert(dst_id);
}

void GraphBuilder::addEdgeTarget(int src_id, int dst_id, const std::vector<double> &target) {
	addEdgeTarget(src_id, dst_id);

	if (target.size() != mNumTargetEdgeFeatures) {
		throw std::invalid_argument("Edge target feature size does not match number of target edge features");
	}
	if (mEdgeTargetFeatures.count({src_id, dst_id}) >= 1) {
		throw std::invalid_argument("Edge target with the same source and destination already has features");
	}
	mEdgeTargetFeatures[{src_id, dst_id}] = target;
}

void GraphBuilder::reset() {
	mNodeFeatures.clear();
	mNodeTargets.clear();
	mEdges.clear();
	mEdgeTargets.clear();
	mEdgeFeatures.clear();
	mEdgeTargetFeatures.clear();
	mBuilt = false;
}

GraphData GraphBuilder::buildGraph() {
	GraphData graphData{
		.nodeIds = {},
		.nodeFeatures = {},
		.nodeTargets = {},
		.edgeFeatures = {},
		.edgeTargetFeatures = {},
		.edgeIndex = {{}, {}},
		.edgeTargetIndex = {{}, {}}
	};

	for (const auto &[id, features] : mNodeFeatures) {
		graphData.nodeIds.push_back(id);
		graphData.nodeFeatures.push_back(features);
		if (mNodeTargets.count(id)) {
			graphData.nodeTargets.push_back(mNodeTargets.at(id));
		} else {
			graphData.nodeTargets.push_back(std::vector<double>{});
		}
	}

	// edges is now just coo
	// no duplicates allowed in the set
	// 1 -> [2,3,4,5]
	// 2 -> [1,2,3]
	// ...

	// build edges
	for (const auto &[src_id, neighbors] : mEdges) {
		for (const auto &dst_id : neighbors) {
			graphData.edgeIndex[0].push_back(src_id);
			graphData.edgeIndex[1].push_back(dst_id);
			// add edge features if exist here so the order matches edgeIndex
			graphData.edgeFeatures.push_back(
				mEdgeFeatures.count({src_id, dst_id}) ? mEdgeFeatures.at({src_id, dst_id}) : std::vector<double>{}
			);
		}
	}

	// build edge targets
	for (const auto &[src_id, targets] : mEdgeTargets) {
		for (const auto &dst_id : targets) {
			graphData.edgeTargetIndex[0].push_back(src_id);
			graphData.edgeTargetIndex[1].push_back(dst_id);
			graphData.edgeTargetFeatures.push_back(
				mEdgeTargetFeatures.count({src_id, dst_id}) ? mEdgeTargetFeatures.at({src_id, dst_id})
															: std::vector<double>{}
			);
		}
	}

	mBuilt = true;
	return std::move(graphData);
}

int GraphBuilder::getNumEdges() const {
	int count = 0;
	for (const auto &[src_id, neighbors] : mEdges) {
		count += neighbors.size();
	}
	return count;
}
int GraphBuilder::getNumEdgeTargets() const {
	int count = 0;
	for (const auto &[src_id, targets] : mEdgeTargets) {
		count += targets.size();
	}
	return count;
}