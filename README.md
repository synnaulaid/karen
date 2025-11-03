# karen

<img src="assets/img/karen.jpg">

Roadmap:

- Load dataset 1B (initial dataset)
- Train-Model (v1)

Test Mode: `python3 test/test_model.py`

if no support

```
pip uninstall bitsandbytes peft
pip install peft==0.4.0

```

Dataset ref: `https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/raw/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
download copy to dataset/

```
git clone https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
cd ShareGPT_Vicuna_unfiltered
git lfs pull
```
and copy dataset to dir dataset/ ex: `cp ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json /path/to/project/dataset/`

