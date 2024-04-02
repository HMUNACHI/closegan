# CloseGAN: Conditional Latent-Optimised Sequence GAN For Creative Text Generation.
(please review the paper to see architecture details and results)

# Abstract 
Computational humour is a difficult and overlooked challenge in natural language processing, yet humour remains an integral part of human communication and a recipe for interesting discourse. While considerable progress has been made in language synthesis, the nuanced task of humour generation transcends conventional syntactic accuracy and semantic sanity. Hilarity and originality are key, and humour is subjective. This investigation exposes the intricacies of humour generation; It underscores that humour's essence is underpinned by purposeful disjointed patterns, including the presence of uncommon tokens and contextual dissonance, with manifestations akin to noise that contravene semantic cohesion. Conventional language models struggle to learn these weak but indispensable comedic patterns. During inference, the likelihoods of these rare patterns are inconsequential for generation strategies like Beam Search and Top-K Sampling to ensure their inclusion. To address this, we propose a novel approach: replacing the encoder in a sequence-to-sequence setup with a conditional stochastic latent-sampling encoder. This modification better mimics human creativity, accommodating the incorporation of these infrequent yet impactful patterns. Our results demonstrate a competitive level of hilarity when measured against human-curated content. Furthermore, our technique showcases heightened creativity and originality in contrast to existing methods.

# Authors
Henry Ndubuaku\
ndubuakuhenry@gmail.com\
[Linkedin](https://www.linkedin.com/in/henry-ndubuaku-7b6350b8/)\
[Paper](https://www.researchgate.net/publication/373111729_Improving_Creativity_In_Humour_Generation_With_Conditional_Stochastic_Latent-Sampling_Encoders)
