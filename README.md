# PaliGemma
Following along with the one and only Umar Jamil: https://www.youtube.com/watch?v=vAmKB7iPkWw

## Notes:
- Why do we use contrastive learning when training VMs? :
  - Because we not only want our model to learn good representations for the images/text, but also to have it learn some mutual representation between the two.
  - There's also the fact of data availibility (so many pairs of images that have text asociated with them)
- What was the motivation behind using siglip as loss function:
  - The softmax was computationnal heavy, we had a matrix that results from the dot product of shape N by N, we needed to do:
    - Row wise softmax
    - AND Column wise softmax, because there's no symetry we can benifit from (dot a times b != dot b times a)
