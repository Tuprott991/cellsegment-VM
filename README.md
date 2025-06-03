### How this model solve the cell segmentation task
- 1. Seek suitable seed pixels to represent each object (cell)
- 2. Compute a similarity metric to relate each pixel to each seed pixel
- 3. Use a neural network to map a similarity metric, conditioned on learnt higher object properties, to a probability
of belonging to an object.