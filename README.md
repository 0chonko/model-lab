# LLM-basics
 Practice directory to understand the underlying mechanisms of a multi-modal genAI model

## References

- Deiseroth et al., [T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings](https://arxiv.org/abs/2406.19223).

## Course labs

Follow the exercises from attention and memory through model adaptation and structured extraction. Each lab has a small example, a question to investigate, and a way to check the result. The [experiment notes](studies/README.md) record what changed across runs and where the results stopped generalizing.

Start with the existing [tokenization examples](Text%20Encoding/) and [PyTorch and attention exercises](Transformers/). Then work through the [lab guide](labs/README.md).

```sh
python -m pip install -r requirements-course.txt
OMP_NUM_THREADS=1 python -m pytest -q
```

The default test suite runs on CPU with synthetic inputs. The Qwen, LFM2, and embedding exercises use [optional model dependencies](requirements-models.txt) and download weights on first use.
