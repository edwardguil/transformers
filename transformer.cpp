#include "transformer.h"

Attention::Attention(std::size_t in_dims, std::size_t n_heads, AttentionType type_)
    : query(register_module("query", torch::nn::Linear(in_dims, n_heads))),
      key(register_module("key", torch::nn::Linear(in_dims, n_heads))),
      value(register_module("value", torch::nn::Linear(in_dims, n_heads))),
      type(type_) {}

torch::Tensor Attention::forward(torch::Tensor x, std::optional<torch::Tensor> context) {
    torch::Tensor k, v;
    torch::Tensor q = query(x); // BxTxH

    if (type == AttentionType::Cross && context.has_value()) {
        k = key(context.value()); // BxTxH
        v = value(context.value()); // BxTxH
    } else {
        k = key(x); // BxTxH
        v = value(x); // BxTxH
    }

    torch::Tensor attn_scores = torch::matmul(q, torch::transpose(k, -2, -1)); // BxTxH @ BxHxT = BxTxT

    if (type == AttentionType::Casual) {
        auto mask = torch::tril(torch::ones({attn_scores.size(-1), attn_scores.size(-1)})); // BxTxT
        attn_scores = attn_scores.masked_fill(mask == 0, -std::numeric_limits<float>::infinity()); // BxTxT
    }

    attn_scores = torch::nn::functional::softmax(attn_scores, torch::nn::functional::SoftmaxFuncOptions(1));

    torch::Tensor wei_sum = torch::matmul(attn_scores, v); // BxTxT @ BxTxH = BxTxH

    return wei_sum;
}

MLP::MLP(std::size_t n_embeds, std::size_t hid_scale)
    : dropout(register_module("dropout", torch::nn::Dropout())) {
    layers.push_back(register_module("layer1", torch::nn::Linear(n_embeds, n_embeds * hid_scale)));
    layers.push_back(register_module("layer2", torch::nn::Linear(n_embeds * hid_scale, n_embeds)));
}

torch::Tensor MLP::forward(torch::Tensor x) {
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        x = torch::relu(layers[i]->forward(x));
    }
    x = layers.back()->forward(x);
    x = dropout(x);
    return x;
}

EncoderBlock::EncoderBlock(std::size_t n_embeds, std::size_t n_heads)
    : attention(n_embeds, n_heads, Attention::AttentionType::Self),
      mlp(n_embeds),
      layer_norm(register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{static_cast<int64_t>(n_embeds)}))) {}

torch::Tensor EncoderBlock::forward(torch::Tensor x) {
    torch::Tensor out = attention.forward(x);
    x = x + layer_norm(out); // Add & Norm
    out = mlp.forward(x);
    x = x + layer_norm(out); // Add & Norm

    return x;
}

DecoderBlock::DecoderBlock(std::size_t n_embeds, std::size_t n_heads, bool decoderOnly_)
    : masked_attention(n_embeds, n_heads, Attention::AttentionType::Casual),
      cross_attention(n_embeds, n_heads, Attention::AttentionType::Cross),
      mlp(n_embeds),
      layer_norm(register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{static_cast<int64_t>(n_embeds)}))),
      decoderOnly(decoderOnly_) {}

torch::Tensor DecoderBlock::forward(torch::Tensor x, std::optional<torch::Tensor> e_output) {
    torch::Tensor out = masked_attention.forward(x);
    x = x + layer_norm(out); // Add & Norm
    if (!decoderOnly) {
        out = cross_attention.forward(x, e_output);
        x = x + layer_norm(out);
    }
    out = mlp.forward(x);
    x = x + layer_norm(out); // Add & Norm

    return x;
}

Transformer::Transformer(std::size_t n_embeds, std::size_t block_size,
                         std::size_t e_vocab_size, std::size_t num_blocks, std::size_t n_heads,
                         std::size_t output_size)
    : e_tok_embed_table(register_module("e_tok_embed_table", torch::nn::Embedding(e_vocab_size, n_embeds))),
      e_pos_embed_table(register_module("e_pos_embed_table", torch::nn::Embedding(block_size, n_embeds))),
      d_tok_embed_table(register_module("d_tok_embed_table", torch::nn::Embedding(e_vocab_size, n_embeds))),
      d_pos_embed_table(register_module("d_pos_embed_table", torch::nn::Embedding(block_size, n_embeds))),
      last_linear(register_module("last_linear", torch::nn::Linear(n_embeds, output_size))) {
    for (std::size_t i = 0; i < num_blocks; ++i) {
        encoder_blocks.emplace_back(n_embeds, n_heads);
        decoder_blocks.emplace_back(n_embeds, n_heads);
    }
}

torch::Tensor Transformer::forward(torch::Tensor x, torch::Tensor y) {
    torch::Tensor e_tok_embed = e_tok_embed_table->forward(x);
    torch::Tensor e_pos_embed = e_pos_embed_table->forward(torch::arange(x.size(1)));

    torch::Tensor d_tok_embed = d_tok_embed_table->forward(y);
    torch::Tensor d_pos_embed = d_pos_embed_table->forward(torch::arange(y.size(1)));

    torch::Tensor e_embed = e_tok_embed + e_pos_embed;
    torch::Tensor d_embed = d_tok_embed + d_pos_embed;

    for (std::size_t i = 0; i < encoder_blocks.size(); ++i) {
        e_embed = encoder_blocks[i].forward(e_embed);
        d_embed = decoder_blocks[i].forward(d_embed, e_embed);
    }

    torch::Tensor out = torch::nn::functional::softmax(last_linear(d_embed), torch::nn::functional::SoftmaxFuncOptions(1));

    return out;
}

int main() {
	return 0;
}