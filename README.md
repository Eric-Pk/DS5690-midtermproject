# Extending Context Window of Large Language Models via Position Interpolation

## Authors

- Shouyuan Chen
- Sherman Wong
- Liangjian Chen
- Yuandong Tian

**Affiliation**: Meta Platforms Inc.
![image text](https://github.com/Eric-Pk/DS5690-midtermproject/blob/main/outline.png)
## Overview

### What problem does this article solve?

  This paper introduces a technique called Position Interpolation that can expand the context window size of RoPE-based pretrained LLMs with minimal fine-tuning, thereby achieving strong empirical results on a variety of tasks requiring long context, including password retrieval, language modeling and long document summarization.
  
### What are the shortcomings of existing solutions?

  A shortcoming of the existing solutions mentioned in this article is their inability to handle long sequences. Specifically, existing Transformer models usually need to divide the input sequence into fixed-length chunks and then process these chunks separately. This method can handle short sequences, but for long sequences, it will cause the performance of the model to decrease. To solve this problem, some recent research has proposed some new methods, such as extending the representation of short sequences to long sequences (Length Extrapolation), or using more efficient attention mechanisms to process long sequences (such as Adaptive Attention). However, these methods still have some limitations, such as the need to increase model parameters or high computational complexity, and thus cannot handle very long sequences. In order to further improve the performance of the model, this paper proposes a new technique called Position Interpolation, which can handle longer sequences without increasing model parameters. Specifically, Position Interpolation uses a position encoding-based technology called Linear Position Interpolation to expand the context window. By inserting additional position encoding, Position Interpolation can extend the context window to handle longer sequences without increasing model parameters. Compared with other technologies for extending the context window, Position Interpolation has better performance and higher efficiency.

### Solution

  The solution in this article is to extend the context window of the Transformer model to handle longer sequences through a technique called Position Interpolation. Specifically, Position Interpolation uses a position encoding-based technology called Linear Position Interpolation to expand the context window. The basic idea of Linear Position Interpolation is to insert some additional position coding based on the original position coding to expand the context window. Specifically, for each position, Linear Position Interpolation calculates the distance between that position and other positions and inserts some additional position encoding based on these distances. These additional position codes can be computed by linear interpolation, allowing them to be combined with the original position codes to form a longer context window. In addition to Linear Position Interpolation, this article also uses some other techniques to further improve the performance of the model. For example, this article uses a positional encoding technique called Relative Positional Encoding, which can better capture local structure and long-range dependencies in sequences. In addition, this paper also uses an attention mechanism called Adaptive Attention, which can adaptively handle long sequences. The solution in this article has achieved good performance in multiple natural language processing tasks, such as GLUE, SuperGLUE and SQuAD. Compared with other techniques for extending the context window, our solution has better performance and higher efficiency because it can handle longer sequences without increasing model parameters.

## Architecture Overview

### What is RoPE-based technology? Introduce in detail

  RoPE is the abbreviation of Rotary Position Embedding, which is a position encoding technology used to inject explicit position information into the Transformer model to represent the order of input. RoPE is the position encoding method used in the LLaMA model. The Position Interpolation technology introduced in this article can expand the context window size of RoPE-based pre-trained LLMs.
  Rotary Position Embedding (RoPE) is a position encoding technique used to inject explicit position information into the Transformer model to represent the order of inputs. RoPE was proposed by Jianlin Su et al. in 2021 to solve some problems of traditional position encoding methods. The main idea of RoPE is to treat position encoding as a rotating phase angle rather than a static vector. This rotated phase angle can be achieved by using a rotation matrix, allowing RoPE to be extended to longer sequence lengths without increasing model parameters. Specifically, RoPE represents the position encoding as a complex vector, where each element is a complex number representing a phase angle of rotation. For each position, RoPE uses a different rotation matrix to calculate the corresponding position encoding. This rotation matrix can be achieved by using Givens rotation, allowing RoPE to scale to longer sequence lengths without increasing model parameters. The advantage of RoPE is that it can be extended to longer sequence lengths and can be implemented without increasing model parameters. In addition, RoPE can also implement positional encoding by using complex number operations, making positional encoding more flexible and interpretable. RoPE has been proven to perform well in various natural language processing tasks.


### In LLM that is not RoPE-based pre-training, what is SOTA's approach to extending the context window?
  According to the description in 1, the current SOTA approach is to use some special technologies, such as ALiBi and LeX, to implement length extrapolation of the Transformer, that is, to expand the context window without increasing model parameters. These techniques enable the model to handle longer sequences by introducing some additional parameters and structures into the model. However, the effects of these techniques still await further research and exploration. In addition, 1 also mentioned a method to directly fine-tune the pre-trained model, but the effect of this method is not ideal because the model has difficulty adapting to a longer context window.

### Introduce ALiBi in detail

  ALiBi is a technology used to expand the context window of the Transformer model. Its full name is Adaptive Long-range Big model inference. The main idea of ALiBi is to handle long sequences in an adaptive manner, rather than simply increasing the parameters or structure of the model. Specifically, ALiBi uses an adaptive attention mechanism, called Adaptive Attention, to handle long sequences. Adaptive Attention divides the sequence into chunks and assigns a different attention weight to each chunk. These weights are calculated through an adaptive mechanism that automatically adjusts based on the length of the sequence and the parameters of the model. In this way, ALiBi can expand the context window without increasing model parameters, thereby processing longer sequences. In addition to Adaptive Attention, ALiBi also uses some other technologies to further improve the performance of the model. For example, ALiBi uses a positional encoding-based technology called Relative Positional Encoding to represent the relative distance between different positions in the sequence. This encoding method can better capture the local structure and long-range dependencies in the sequence. ALiBi has achieved good performance in multiple natural language processing tasks, such as GLUE, SuperGLUE and SQuAD. Compared with other techniques that extend the context window, ALiBi has better performance and higher efficiency because it can handle longer sequences without increasing model parameters.

### Introduce LeX in detail
  LeX is a technology for extending the context window of the Transformer model, and its full name is Longformer with Extra attention. The main idea of LeX is to handle long sequences by introducing an additional attention mechanism, rather than simply increasing the parameters or structure of the model. Specifically, LeX uses an attention mechanism called Local-Global Attention, which can handle both local and global information. Local-Global Attention divides the sequence into chunks and assigns a different attention weight to each chunk. These weights are calculated through an adaptive mechanism that automatically adjusts based on the length of the sequence and the parameters of the model. In this way, LeX can extend the context window without increasing the model parameters and thus handle longer sequences. In addition to Local-Global Attention, LeX also uses some other technologies to further improve the performance of the model. For example, LeX uses a position encoding technique called Rotary Position Embedding (RoPE), which can extend sequence length without increasing model parameters. RoPE represents the position encoding as a complex vector, where each element is a complex number representing a phase angle of rotation. For each position, RoPE uses a different rotation matrix to calculate the corresponding position encoding. This rotation matrix can be achieved by using Givens rotation, allowing RoPE to scale to longer sequence lengths without increasing model parameters. LeX has achieved good performance in multiple natural language processing tasks, such as GLUE, SuperGLUE and SQuAD. Compared to other techniques for extending the context window, LeX has better performance and higher efficiency because it can handle longer periods without increasing model parameters.

### Detailed introduction to the Position Interpolation technology in this article’s solution
Position Interpolation is a technique proposed in this article for extending the context window of the Transformer model. The basic idea is to insert some additional position encoding based on the original position encoding to expand the context window.

Specifically, Position Interpolation uses a position encoding-based technology called Linear Position Interpolation to expand the context window. The basic idea of Linear Position Interpolation is to insert some additional position coding based on the original position coding to expand the context window.

For each position, Linear Position Interpolation calculates the distance between that position and other positions and inserts some additional position encoding based on these distances. These additional position codes can be computed by linear interpolation, allowing them to be combined with the original position codes to form a longer context window.

Assume that the original position encoding is $P_1, P_2, ..., P_n$, where $n$ is the length of the original sequence. In order to expand the context window, Position Interpolation will insert some additional position codes $P_{n+1}, P_{n+2}, ..., P_m$, where $m$ is the extended sequence length. These additional position codes can be calculated by linear interpolation.

For each additional position $i$, its position encoding can be calculated as: $$P_i = \frac{i-n}{m-n}P_m + \frac{m-i}{m-n}P_n$$ where, $i$ represents additional positions, $m$ represents the extended sequence length. In this way, Position Interpolation can extend the context window to handle longer sequences without increasing model parameters.

### Experiment

This article conducted experiments on two data sets, namely PG-19 and Arxiv Math proof-pile. Among them, PG-19 is a corpus containing 50,000 English books, and Arxiv Math proof-pile is a corpus containing mathematical proofs. In the experiment, this article uses two evaluation indicators, namely perplexity and ROUGE-2 score. Perplexity is a metric used to evaluate the performance of a language model. It measures how difficult it is for the model to predict the next word given a sequence. The ROUGE-2 score is a metric used to evaluate the quality of text summarization, which measures the similarity between model-generated summaries and human-generated summaries. In the experiment, this article used two methods to expand the context window of the Transformer model, namely Position Interpolation and Direct Fine-tuning. For each method, this paper uses two different positional encoding techniques, namely Relative Positional Encoding and Linear Position Interpolation. By comparing the performance of different methods and different position encoding techniques, this paper demonstrates the effectiveness and efficiency of Position Interpolation.

In the experiment, this article compared the following four baseline models:
1. Vanilla Transformer: This is a standard Trans
Position Interpolation (PI) extends the context window sizes of RoPE-based pretrained LLMs. The problem addressed is the limitation of context window sizes in existing models, and the solution presented is the introduction of Position Interpolation to overcome this limitation. The approach demonstrates strong empirical results on various tasks requiring long context.
2. Length Extrapolation: This is a technique that extends the representation of short sequences to long sequences by adding additional positional encoding at both ends of the sequence to expand the context window.
3. Adaptive Attention: This is a technique that uses a more efficient attention mechanism to process long sequences by using a learnable gating mechanism in the attention calculation to selectively focus on different locations.
4. Direct Fine-tuning: This is a technology that directly fine-tunes the model to adapt to specific tasks by performing supervised fine-tuning on the basis of a pre-trained model.

Experimental results show that the Position Interpolation method proposed in this article achieves the best performance on both data sets. Specifically, on the PG-19 data set, the perplexity of the model using the Position Interpolation method under the 8192 context window is 7.13, which is lower than the model using other methods. On the Arxiv Math proof-pile data set, the ROUGE-2 score of the model using the Position Interpolation method under the 8192 context window is 0.305, which is higher than the model using other methods. In addition, this article also found that the model using the Position Interpolation method has achieved good performance under different position encoding technologies, especially when using the Linear Position Interpolation technology, the performance of the model is even better. This shows that the Position Interpolation method is very versatile and robust and can be applied to different position encoding technologies and different tasks.

Ablation experiments show that using the Position Interpolation method to improve model performance mainly comes from two aspects: one is the ability to expand the context window, and the other is better utilization of long sequence information. Specifically, this article conducted two sets of ablation experiments, namely removing the Position Interpolation method and removing Linear Position Interpolation technology. The results show that after removing the Position Interpolation method, the model's perplexity on the PG-19 data set increased from 7.13 to 7.21. After removing the Linear Position Interpolation technology, the model's perplexity on the PG-19 data set increased from 7.13 to 7.23. This shows that the improvement of model performance by the Position Interpolation method mainly comes from the ability to expand the context window, while the contribution of Linear Position Interpolation technology is relatively small.

## Critical Analysis

1.The performance of the Position Interpolation method is tested on different training data sets to evaluate its performance on different context corpora.

2.The Position Interpolation method can be applied on multiple language models, such as testing on GPT, Transformer, and other recurrent neural network-based language models.

3.Other improvement methods can be added to the Position Interpolation method, such as adding other training examples during the training process to improve model performance.

4.Improvements can be made in the architecture of the training model, such as introducing long-tone training in the attention mechanism to improve model performance. The above information mainly comes from pages 9, 10, and 1

## Questions for Discussion

1. Question 1: what aspect do you think this methodology can be improved in the future?
2. Question 2: what is the limitation for this methodology?

## Resources

- [[Link to the paper](https://arxiv.org/abs/2306.15595)](#)
- [[Link to a related tutorial post](https://www.youtube.com/watch?v=oyXdmtHgZFw&t=53s)](#)
- [[PapersWithCode link](https://github.com/ymcui/chinese-llama-alpaca-2)](#)


## Citation

@article{chen2023extending,
  title={Extending context window of large language models via positional interpolation},
  author={Chen, Shouyuan and Wong, Sherman and Chen, Liangjian and Tian, Yuandong},
  journal={arXiv preprint arXiv:2306.15595},
  year={2023}
}
