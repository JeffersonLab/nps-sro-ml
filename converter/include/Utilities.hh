#ifndef UTILITIES_HH
#define UTILITIES_HH

#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

// Undefine the ROOT macro that conflicts with torch
#ifdef ClassDef
#undef ClassDef
#endif

#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>

template <typename T> std::vector<T> flatten(const std::vector<std::vector<T>> &input) {
	std::vector<T> output;
	for (const auto &vec : input) {
		output.insert(output.end(), vec.begin(), vec.end());
	}
	return output;
}

template <typename T> torch::Tensor toTensor(const std::vector<T> &data) {

	auto options = torch::TensorOptions().device(torch::kCPU);

	if constexpr (std::is_same_v<T, float>)
		options = options.dtype(torch::kFloat32);
	else if constexpr (std::is_same_v<T, double>)
		options = options.dtype(torch::kFloat32);
	else if constexpr (std::is_same_v<T, int32_t>)
		options = options.dtype(torch::kInt32);
	else if constexpr (std::is_same_v<T, int64_t>)
		options = options.dtype(torch::kInt32);
	else
		static_assert(!sizeof(T *), "Unsupported data type for toTensor()");

	return torch::from_blob(const_cast<T *>(data.data()), {static_cast<int64_t>(data.size())}, options).clone();
}

template <typename T> torch::Tensor toTensor2D(const std::vector<std::vector<T>> &data) {
	auto options = torch::TensorOptions().device(torch::kCPU);

	if constexpr (std::is_same_v<T, float>)
		options = options.dtype(torch::kFloat32);
	else if constexpr (std::is_same_v<T, double>)
		options = options.dtype(torch::kFloat64);
	else if constexpr (std::is_same_v<T, int32_t>)
		options = options.dtype(torch::kInt32);
	else if constexpr (std::is_same_v<T, int64_t>)
		options = options.dtype(torch::kInt64);
	else
		static_assert(!sizeof(T *), "Unsupported data type for toTensor2D()");

	int64_t n_rows = data.size();
	int64_t n_cols = n_rows > 0 ? data[0].size() : 0;

	auto flat = flatten(data);
	return torch::from_blob(flat.data(), {n_rows, n_cols}, options).clone();
}

template <typename... Tensors> void saveTensors(const std::string &output_file, const Tensors &...tensors) {
	// Static check: all arguments must be torch::Tensor
	static_assert(
		(std::conjunction_v<std::is_same<Tensors, torch::Tensor>...>),
		"All arguments to saveTensors() must be torch::Tensor"
	);

	// Collect tensors into a vector
	std::vector<torch::Tensor> tensor_vec{tensors...};

	// Serialize with Torch pickle
	torch::IValue ivalue(tensor_vec);
	std::vector<char> buffer = torch::pickle_save(ivalue);

	// Write to binary file
	std::ofstream fout(output_file, std::ios::binary);
	if (!fout) {
		throw std::runtime_error("Failed to open file: " + output_file);
	}
	fout.write(buffer.data(), buffer.size());
	return;
}

#endif