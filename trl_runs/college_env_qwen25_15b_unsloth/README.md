---
base_model: sshleifer/tiny-gpt2
library_name: transformers
model_name: college_env_qwen25_15b_unsloth
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for college_env_qwen25_15b_unsloth

This model is a fine-tuned version of [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- TRL: 1.2.0
- Transformers: 4.57.2
- Pytorch: 2.11.0
- Datasets: 4.8.4
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and GallouÃ©dec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```