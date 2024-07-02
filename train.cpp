#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "transformer.h"

std::pair<torch::Tensor, torch::Tensor> load_dataset(const std::string& input_file, const std::string& target_file, std::size_t vocab_size, std::size_t seq_length) {
    std::ifstream input_stream(input_file);
    std::ifstream target_stream(target_file);
    std::vector<std::vector<int>> input_data;
    std::vector<std::vector<int>> target_data;
    std::string input_line, target_line;

    while (std::getline(input_stream, input_line) && std::getline(target_stream, target_line)) {
        std::istringstream input_iss(input_line);
        std::istringstream target_iss(target_line);
        int input_token, target_token;

        std::vector<int> input_sequence;
        std::vector<int> target_sequence;

        while (input_iss >> input_token && target_iss >> target_token) {
            input_sequence.push_back(input_token);
            target_sequence.push_back(target_token);
        }

        // Pad sequences
        if (input_sequence.size() < seq_length) {
            input_sequence.resize(seq_length, 0);
            target_sequence.resize(seq_length, 0);
        }

        input_data.push_back(input_sequence);
        target_data.push_back(target_sequence);
    }

    torch::Tensor input = torch::zeros({static_cast<int>(input_data.size()), static_cast<int>(seq_length)}, torch::kInt64);
    torch::Tensor target = torch::zeros({static_cast<int>(target_data.size()), static_cast<int>(seq_length)}, torch::kInt64);

    for (size_t i = 0; i < input_data.size(); ++i) {
        for (size_t j = 0; j < input_data[i].size(); ++j) {
            input[i][j] = input_data[i][j];
            target[i][j] = target_data[i][j];
        }
    }

    // Offset target by 1
    target = torch::cat({target.slice(1, 0, -1), torch::zeros({target.size(0), 1}, torch::kInt64)}, 1);

    return {input, target};
}

int main() {
    // Model parameters
    const std::size_t batch_size = 5;
    const std::size_t num_epochs = 10;

    const std::size_t n_embeds = 256;
    const std::size_t n_heads = 8;
    const std::size_t num_layers = 12;
    const std::size_t block_size = 1024;

    // Data parameters
    const std::string input_file = "integer_encoded/train.de"; // x set
    const std::string target_file = "integer_encoded/train.en"; // y set
    const size_t seq_length = 64;
    const std::size_t vocab_size = 10000;
    auto [input, target] = load_dataset(input_file, target_file, vocab_size, seq_length);

    // Model setup
    Transformer model = Transformer(n_embeds, block_size, vocab_size, num_layers, n_heads, vocab_size);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
    torch::nn::CrossEntropyLoss criterion;

    float total_loss = 0.0;
    model.train();
    for (std::size_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = 0; i < input.size(0) - batch_size; i += batch_size) {
            auto x = input.slice(0, i, i+batch_size);
            auto y = target.slice(0, i, i+batch_size);

            optimizer.zero_grad();
            auto output = model.forward(x, y);
            auto loss = criterion(output.view({-1, vocab_size}), y.view({-1}));
            loss.backward();
            optimizer.step();
            
            total_loss = total_loss + loss.item<float>();
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: "
                        << std::fixed << std::setprecision(4) << total_loss / ((i + batch_size) / batch_size)
                        << "\r" << std::flush;
        }

        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: "
            << std::fixed << std::setprecision(4) << total_loss / ((input.size(0) - 1) / batch_size + 1)
            << std::endl;
        total_loss = 0.0;
    }

    std::cout << "Training completed!" << std::endl;

    return 0;
}
