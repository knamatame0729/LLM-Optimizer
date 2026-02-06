# LLM-Based Numerical Optimizer
An optimization framework that uses Large Language Models (LLMs) to find global minima of functions through iterative reasoing and exploration.

## Overview
This project demonstraets how LLMs can be used as optimizers for numerical optimization problems through in-context learning. The LLM acts as an intelligent agent that:

- Analyzes previous function evaluations provided in the context
- Reasons about the function landscape through in-context examples
- Propose new paramter values to explore
- Behave as gradient-free numerical optimizers.

## Optimization Objective

The goal is to find:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})
$$

where:

- $\mathbf{x} \in \mathbb{R}^d$ is the parameter vector
- $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is the objective function to minimize
- $\mathcal{X} = [x_{\min}, x_{\max}]$ is the parameter space


## Features
- Uses LLMs ability(In-Context Learning) to learn from examples in the prompt without fine-tuning
- Uses LLM for parameter proposals.
- No derivative information

## How It Works
### In-Context Learning Mechanism
The optimizer leverages in-context learning - the LLM's ability to learn patterns from examples provided in the prompt without any parameter updates or fine-tuning. At each iteration:
1. The LLM receives a history of (parameter, reward) pairs as examples
2. It identifies patterns and trends purely from these in-context examples
3. It proposes new parameters based on this learned understanding
4. No gradient computation or model weights modification occurs