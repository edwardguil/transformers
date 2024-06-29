#include <torch/torch.h>
#include <memory>
#include <iostream>
#include <string>

// For now, omit using namespaces for clarity

class Attention : torch::nn::Module {
  public:
    enum class AttentionType {
      Self,
      DotProduct
    };

  private:
    torch::nn::Linear query;
    torch::nn::Linear key;
    torch::nn::Linear value;
    AttentionType type;

  public:

    Attention (std::size_t in_dims, std::size_t heads, AttentionType type_) {
      query = torch::nn::Linear(in_dims, heads);
      key = torch::nn::Linear(in_dims, heads);
      value = torch::nn::Linear(in_dims, heads);
      type = type_;
    }

    torch::Tensor forward(torch::Tensor input) {
      torch::Tensor q = query(input);
      torch::Tensor k = key(input);
      torch::Tensor v = value(input);

      torch::Tensor attn_scores = torch::matmul(q, torch::transpose(k, -2, -1));
      
      if (type == AttentionType::Self) {
        // Do the triangular lower masking trick
      }
      
      torch::Tensor attn_scores = torch::nn::functional::softmax(attn_scores, torch::nn::functional::SoftmaxFuncOptions(1));

      torch::Tensor wei_sum = torch::matmul(attn_scores, v);

      return wei_sum;
    }


};

class Transformer : public torch::nn::Module {
    std::unique_ptr<torch::nn::Embedding> tok_embed_table; 
    std::unique_ptr<torch::nn::Embedding> pos_embed_table;

    public:
      Transformer(std::size_t n_embeds, std::size_t block_size,  
        std::size_t vocab_size) {
        tok_embed_table = std::make_unique<torch::nn::Embedding>((vocab_size, n_embeds));
        pos_embed_table = std::make_unique<torch::nn::Embedding>((block_size, n_embeds));
      }

      torch::Tensor foward(torch::Tensor x, torch::Tensor y) {
        torch::Tensor tok_embed = (*tok_embed_table)(x);
        torch::Tensor pos_embed = (*pos_embed_table)(x);

        torch::Tensor embed = tok_embed + pos_embed;

        // We've got our joint embeds, it's time for some self attention

        return torch::rand({2, 3});
      }

};


int main() {
  std::cout << "Not yet" << std::endl;
}

