# Universal Transformer U-Net
This branch changes the architecture into a universal-transformer-style U-Net, which should allow us to throw more compute at the problem without increasing param count/information capacity and needing heavy regularization (like the super high weight decay in the baseline).

Instead of giving every layer its own full attention and MLP weights, there is one encoder and one decoder block (attn/MLP layers only) shared across depth. Each depth still keeps its own norms, gates, residual scalars, skip weight, and value projection to keep expressiveness. The norms have biases that work a bit like a learned depth embedding. (The depth is restricted to an even number for this to work)

I also increased the MLP multiplier instead of only increasing model width to get to the desired param count. The intuition is that the MLP neurons can be seen as a collection of key-value pairs. Simply increasing model width increases param count quadratically while increasing neuron or kv-pair count linearly, so we directly increase neuron count instead. (This view of the MLP layer is detailed more in this bytedance paper which i think is a great read: https://arxiv.org/pdf/2505.19488v1)

The architecture allows us to train with a depth schedule, which speeds up training quite a bit. The model starts shallow and grows deeper over time (adding depth towards the middle, meaning encoder depth is increased at the end and decoder in the beginning).

Parameter count decreases to 512M (260M for the 2 shared blocks of width 2048 with 8x mlp mul, rest is embeddings and lm head).

It's not competitive yet with the baseline, however it could be a strong contender for the unlimited track if the gains from scaling up model size and weight decay hit a wall.

## Training summary: 
Total training time: 58.82m                                                     
Final train loss: 2.796117                                                      
Min val BPB: 1.074007                                                           
Min val Loss: 3.305028                                                          
Total wall time: 3870.76s (64.51m) 

## More on the universal transformer
Introduced by Google back in 2019 (https://arxiv.org/pdf/1807.03819). They also made a recursive version of BERT called ALBERT for parameter efficiency back in 2020. (https://arxiv.org/pdf/1909.11942)

Similar architectures have been effective for reasoning and ARC-AGI. (https://arxiv.org/pdf/2510.04871, https://arxiv.org/pdf/2512.14693)

 Recursion is a good inductive bias for those kinds of problems. Hasn't been explored yet for sample efficiency in language modeling.
