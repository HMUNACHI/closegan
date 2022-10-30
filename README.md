# closegan

Code to the paper: https://drive.google.com/file/d/1xbtyMn73fR8bx6IMe0sXpVuPo88Gy5PS/view

ABSTRACT
Computational humour is a difficult and overlooked challenge in natural language processing, 
yet humour remains an integral part of human communication and a recipe for interesting discourse. 
Excellent text generation techniques exist, but humour generation is concerned with more than yielding accurate sentences. 
The hilarity is the essence and is extremely hard to capture. 
Previous works have either focused on generating jokes with a specific sentence structure or using maximum likelihood estimation,
which yields repetitive and not-so-creative jokes.
Generative Adversarial Networks (GAN) are deep neural architectures built to generate data from different noise signals on each execution. 
There are no limits to the possible number of random noise signals, hence the number of jokes such a network can generate would be infinite. 
We, therefore, introduce a generative adversarial network architecture for creative joke generation.
We show that our architecture generates very original jokes with high syntactic diversity, semantic accuracy,
and hilarity relative to human-curated jokes and jokes generated with existing techniques.

Author: Henry Ndubuaku
Supervisor: Matthew Purver

Topics: Generative Adversarial Networks, Transformers, Variational Seq2Seq, LSTMs, Humour Theory, Wassertein Loss, Kullback-Leibenher Annealing, NLP, NLG
Tools: Python, TensorFlow, Keras, PyTorch, Numpy
Dataset: rJoke Dataset
Traning: Google Colab Pro for experimentation and GCP Cloud Run Job for full training, you can set up on GCP's Vertex AI or Compute Enigne.

Uses: Humour generation, Lyric generation, Textual dataset generation.
