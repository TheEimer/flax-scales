# Does randomness still matter in DL?
Pitch: we had a big reproducibility crisis, but with more optimized architecture, more data and bigger models, do things like seeding still meaningfully influence performance? And if yes, in what way do we need to watch out for it?

## Phase 1: Generate preliminary ResNet scaling results 
Start with ResNet-50 (in examples), ResNet-18 and ResNet-101. First runs with default HPs, then a small RS around the defaults. 5 seeds first? And then we get an initial impression of how much difference HPO/seeding actually makes across size.

Later: add on wide ResNets (these aren't implemented here, though)