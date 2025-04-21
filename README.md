# LCoW: Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents

This is the official implementation of ICLR 2025 paper, **"Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents"** by [Dongjun Lee](https://dgjun32.github.io), [Juyong Lee](https://jylee425.github.io), [Kyuyoung Kim](https://kykim0.github.io), [Jihoon Tack](https://jihoontack.github.io), [Jinwoo Shin](https://alinlab.kaist.ac.kr), [Yee Whye Teh](https://www.stats.ox.ac.uk/~teh/), [Kimin Lee](https://sites.google.com/view/kiminlee).

## Overview
LCoW is a framework for training the module that contextualizes complicated web page, thereby enhancing the decision-making capabilities of LLM agents in web automation. The contextualization module transforms complex web page into a comprehensible and informative format, enabling LLM agents to make more accurate decisions for web navigation. Our training algorithm to train the contextualization module consists of three phases: (i) trajectory collection, (ii) sampling contextualized observations, and (iii) updating the contextualization module. For each observation from the collected trajectories, we generate multiple contextualized observations using the current contextualization module. Each observation is then assigned a reward based on whether a set of LLM agents can accurately predict the correct action given the contextualized observation. Finally, we select the one with the highest reward as the target and train the contextualization module to maximize the likelihood of the target given the original raw observation.

## Setup
To setup the repository, first run
```
git clone https://github.com/dgjun32/lcow_iclr2025.git
cd lcow_iclr2025
```
For experiment in WebShop, follow the [link](webshop/).\
For experiment in WorkArena and WebArena, follow this [link](browsergym).

## Citation
```
@article{lee2025learning,
  title={Learning to contextualize web pages for enhanced decision making by LLM agents},
  author={Lee, Dongjun and Lee, Juyong and Kim, Kyuyoung and Tack, Jihoon and Shin, Jinwoo and Teh, Yee Whye and Lee, Kimin},
  journal={arXiv preprint arXiv:2503.10689},
  year={2025}
}
```