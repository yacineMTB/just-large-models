# just-LLM
Just large language models. Hackable, with as little abstraction as possible. Done for my own purposes, feel free to rip.

Every model should have its own runnable logic. Seperate. Not shared! Each file does A Thing. The adaptibility of huggingface's code is incredibly bad, due to over abstraction and incidental complexity. Therefore, DIY

## Rules:
- The code is the tool. Edit it as you see fit!
- I will not be addressing issues
- Not a single kwargs shall 
- Every edit shall increase progresss towards removing the huggingface dependency
- One model forward pass = one function call. Simple as.

## Current models
```
python ./llama.py
```