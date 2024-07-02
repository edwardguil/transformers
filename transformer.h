#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <torch/torch.h>
#include <optional>

class Attention : public torch::nn::Module {
public:
    enum class AttentionType {
        Casual,
        Self,
        Cross
    };

private:
    torch::nn::Linear query{nullptr};
    torch::nn::Linear key{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::Linear proj{nullptr};
    AttentionType type;

public:
    Attention(std::size_t in_dims, std::size_t n_heads, AttentionType type_);
    torch::Tensor forward(torch::Tensor x, std::optional<torch::Tensor> context = std::nullopt);
};

class MLP : public torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    torch::nn::Dropout dropout{nullptr};

public:
    MLP(std::size_t n_embeds, std::size_t hid_scale = 4);
    torch::Tensor forward(torch::Tensor x);
};

class EncoderBlock : public torch::nn::Module {
    Attention attention;
    MLP mlp;
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};

public:
    EncoderBlock(std::size_t n_embeds, std::size_t n_heads);
    torch::Tensor forward(torch::Tensor x);
};

class DecoderBlock : public torch::nn::Module {
    Attention masked_attention;
    Attention cross_attention;
    MLP mlp;
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};
    torch::nn::LayerNorm layer_norm3{nullptr};
    bool decoderOnly;

public:
    DecoderBlock(std::size_t n_embeds, std::size_t n_heads, bool decoderOnly_ = false);
    torch::Tensor forward(torch::Tensor x, std::optional<torch::Tensor> e_output = std::nullopt);
};

class Transformer : public torch::nn::Module {
    torch::nn::Embedding e_tok_embed_table{nullptr};
    torch::nn::Embedding e_pos_embed_table{nullptr};
    torch::nn::Embedding d_tok_embed_table{nullptr};
    torch::nn::Embedding d_pos_embed_table{nullptr};
    std::vector<EncoderBlock> encoder_blocks;
    std::vector<DecoderBlock> decoder_blocks;
    torch::nn::Linear last_linear{nullptr};

public:
    Transformer(std::size_t n_embeds, std::size_t block_size,
                std::size_t e_vocab_size, std::size_t num_blocks, std::size_t n_heads,
                std::size_t output_size);
    torch::Tensor forward(torch::Tensor x, torch::Tensor y);
};

#endif // TRANSFORMER_H
