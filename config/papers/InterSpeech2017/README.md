

- MERL_DC_2spk: Basic DeepClustering recipe for 2 speakers for wsj-2spk (here called MERL_2spk)
- MERL_DC_ivec_2spk: DeepClustering recipe for 2 speakers for wsj-2spk, where the oracle i-vectors (extracted from 
single speaker speech) are appended to the input. 
- MERL_DC_estivec_2spk: DeepClustering recipe for 2 speakers for wsj-2spk, where the estimated i-vectors (extracted from 
on speech reconstructions from a model with recipe MERL_DC_2spk) are appended to the input. 