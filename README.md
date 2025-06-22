# [SongBloom]: *Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement*

We propose **SongBloom**, a novel framework for full-length song generation that leverages an interleaved paradigm of autoregressive sketching and diffusion-based refinement. SongBloom employs an autoregressive diffusion model that combines the high fidelity of diffusion models with the scalability of language models.
Specifically, it gradually extends a musical sketch from short to long and refines the details from coarse to fine-grained. The interleaved generation paradigm effectively integrates prior semantic and acoustic context to guide the generation process.
Experimental results demonstrate that SongBloom outperforms existing methods across both subjective and objective metrics and achieves performance comparable to the state-of-the-art commercial music generation platforms.

![img](docs/architecture.png)

Demo page:  [https://cypress-yang.github.io/SongBloom_demo](https://cypress-yang.github.io/SongBloom_demo)

ArXiv: [https://arxiv.org/abs/2506.07634](https://arxiv.org/abs/2506.07634)

## Prepare Environments

```bash
conda create -n SongBloom python==3.8.12
conda activate SongBloom

# yum install libsndfile
# pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 # For different CUDA version
pip install -r requirements.txt
```

## Data Preparation

A  .jsonl file, where each line is a json object:

```json
{
	"idx": "The index of each sample", 
	"lyrics": "The lyrics to be generated",
	"prompt_wav": "The path of the style prompt audio",
}
```

One example can be refered to as: [example/test.jsonl](example/test.jsonl)

The prompt wav should be a 10-second, 48kHz audio clip.

The details about lyric format can be found in [docs/lyric_format.md](docs/lyric_format.md).

## Inference

```bash
source set_env.sh

python3 infer.py --input-jsonl example/test.jsonl
```

## TODO List

- [ ] Support Text Description

## Citation

```
@article{ning2025diffrhythm,
  title={{DiffRhythm}: Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation with Latent Diffusion},
  author={Ziqian, Ning and Huakang, Chen and Yuepeng, Jiang and Chunbo, Hao and Guobin, Ma and Shuai, Wang and Jixun, Yao and Lei, Xie},
  journal={arXiv preprint arXiv:2503.01183},
  year={2025}
}
```

## License

SongBloom (codes and weights) is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). 
