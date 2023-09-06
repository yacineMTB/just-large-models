# Just Large Models
Hackable, with as little abstraction as possible. Done for my own purposes, feel free to rip.

Every model should have its own runnable logic. Seperate. Not shared! Each file does A Thing. The adaptibility of huggingface's code is incredibly bad, due to over abstraction and incidental complexity. Therefore, DIY. Right now, I'm still improving it as I have time.

This is designed to rely on HF for the files, and loading of the files. Otherwise, that is where the dependency ends.

## Rules:
- The code is the tool. Edit it as you see fit!
- All h*ggingface imports will be placed in quarantine. Model pass files will contain no references
- I will not be addressing issues
- Not a single kwargs shall be observed
- One model forward pass = one function call. Simple as.

## Current models
```
python ./llama.py
```
