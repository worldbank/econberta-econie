# EconBERTa x Econ-IE
This repository hosts the domain-adapted language model EconBERTa and the annotated dataset Econ-IE.

Additionally, it contains the official implementation of the experiments found in [EconBERTa: Towards Robust Extraction of Named Entities in Economics](https://aclanthology.org/2023.findings-emnlp.774/)

# Introduction to EconBERTa
EconBERTa is a DeBERTa-based language model adapted to the domain of economics. It has been pretrained following the [ELECTRA](https://arxiv.org/abs/2003.10555) approach, using a large corpus consisting of 9,4B tokens from 1,5M economics papers (around 800,000 full articles and 700,000 abstracts). 

# Use EconBERTa in existing code
We release EconBERTa on huffingface's transformers : [TODO:INSERT LINK]

# Econ-IE dataset
ECON-IE consists of 1, 000 abstracts from economics research papers, totalling more than 7, 000 sentences. 1 The abstracts summarize impact evaluation (IE) studies, aiming to measure the causal effects of interventions on outcomes by using suitable statistical methods for causal inference. The dataset is sampled from 10, 000 studies curated by [3ie](https://www.3ieimpact.org/), published between 1990 and 2022, and covering all 11 sectors defined by the [World Bank Sector Taxonomy](https://thedocs.worldbank.org/en/doc/538321490128452070-0290022017/New-Sector-Taxonomy-and-definitions).
