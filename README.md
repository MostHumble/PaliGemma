# PaliGemma

Following along with the one and only Umar Jamil: https://www.youtube.com/watch?v=vAmKB7iPkWw

## Notes

- Why do we use contrastive learning when training VMs? :
  - Because we not only want our model to learn good representations for the images/text, but also to have it learn some mutual representation between the two.
  - There's also the fact of data availibility (so many pairs of images that have text asociated with them).
    - side note: internet is noisy, but scale can help with that.
- What was the motivation behind using siglip as loss function:
  - The softmax was computationnal heavy, we had a matrix that results from the dot product of shape N by N, we needed to do:
    - Row wise softmax.
    - AND Column wise softmax, because there's no symetry we can benifit from (dot a times b != dot b times a).
- What are the steps that an image goes in order to be encoded in a vision transformer? :
  - We first need to partition/split the image into patches say 16 by 16 pixels.
  - We apply convolution operations on the patches to get a shared represention.
  - We flatten the patches to have one vector (1LP?) and the we concat them.
  - We add learned(-ed+able) positionnal information to give a signal about the order.
- Why do we have a config file for the model? :
  - PaliGemma comes in different sizes.
- Why do we use `register_buffer`?
  - The tensors get moved to the device when the model does.
  - The are not included in model.parameters (is even better than setting param.`requires_grad` to `False`, because the optimzer doesn't iter over them! flipping cool!)
  - It can also be included in the state_dict, except if `persistent` is set to `False`.
- Why do we need positionnal embeddings?
  - It relates to how the positionnal information is lost because the attention scores (from att mech) are invarient to the positions, thus we need a signal to fix this. (note: RNNs like model do not have this problem)
- What is internal coavariate shift?
  - I understand it as follows: during training the input statistics of a batch (mean, var) can change drastically (ex: pics of a desert vs of a sea), this leads to a drastic change in the activations, thus the gradient with vary a lot thus the loss too, hence the gradients too, this seems not to be good proprety for learning.
- How does batch norm work? and how does it attempt to fix? what are its limitations?
  - It work by tracking the statistics at the batch level (bsz, hidden_dim -> 1, hidden_dim).
  - By tracking these metrics (mean, var) we can keep a normal distrubtion of the batch activations, and thus have stable gradients
  - The limitations have to do with batch size, it works better with a bigger batch size, because the statistics are more representative.
- How does layer norm avoid the falls of batch norm?
  - It does so by calculating the stats at the input level (bsz, hidden_dim -> bsz, 1), it's thus independent of the batch size
