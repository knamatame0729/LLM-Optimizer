import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Tuple
from jinja2 import Template
from collections import deque
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

import wandb
from wandb_log import init_wandb, log_iteration, finish_wandb, set_test_prefix

class EpisodeRewardBufferNoBias:
    """A simple replay buffer to store (params, reward) tuples without bias correction."""
    
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, params, reward):
        self.buffer.append((params, reward))
    
    def sort(self):
        self.buffer = deque(sorted(self.buffer, key=lambda x: x[1], reverse=False), maxlen=self.buffer.maxlen)
    
    def __str__(self):
        buffer_table = "Parameters | Reward\n"
        for weights, reward in self.buffer:
            buffer_table += f"{weights[0]:.1f} | {reward:.4f}\n"
        return buffer_table

class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template = None,
        llm_model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        self.llm_model_name = llm_model_name
        self.template_vars = {}

        # Load model
        print(f"Loading model: {llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self.model.eval()
        print("Model loaded!")
        self.model_group = "local"

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})

    def query_llm(self):
        """Query local LLM model"""
        messages = [
            {"role": "system", "content": "You are a numerical optimizer. Always respond in valid JSON format."},
        ]

        messages.extend(self.llm_conversation)

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_token_count = inputs["input_ids"].shape[1]

        print(f"LLM input tokens: {input_token_count}, output tokens: {outputs.shape[1] - input_token_count}")

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Add response to conversation
        self.add_llm_conversation(response, role="assistant")

        return response

    def parse_parameters(self, parameters_string):
        """Parse parameters from LLM response string"""
        new_parameters_list = []

        try:
            print(f"\n=== Parsing LLM Response ===")
            print(f"Raw response: {parameters_string[:500]}...")
            
            # Remove markdown code blocks
            cleaned = parameters_string.strip()
            if '```' in cleaned:
                code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
                if code_match:
                    cleaned = code_match.group(1).strip()
            
            # Try to extract JSON object
            json_match = re.search(r'\{[\s\S]*?\}', cleaned)
            if json_match:
                json_str = json_match.group()
                print(f"Extracted JSON: {json_str[:200]}...")
                
                try:
                    data = json.loads(json_str)
                    if "params" in data or "param" in data:
                        params_data = data.get("params") or data.get("param")
                        
                        # Handle different formats
                        if isinstance(params_data, (int, float)):
                            # Single number
                            new_parameters_list = [float(params_data)]
                        elif isinstance(params_data, str):
                            # String like "1.5" or "-2.4"
                            # Remove brackets if present
                            params_str = params_data.replace('[', '').replace(']', '').strip()
                            # Try to parse as single number first
                            try:
                                param = float(params_str)
                                new_parameters_list = [param]
                            except ValueError:
                                # If not a single number, try comma-separated
                                params_list = []
                                for x in params_str.split(','):
                                    x = x.strip()
                                    if x:
                                        try:
                                            params_list.append(float(x))
                                        except ValueError:
                                            continue
                                if params_list:
                                    new_parameters_list = params_list
                        elif isinstance(params_data, list):
                            # Array
                            params_list = []
                            for x in params_data:
                                try:
                                    params_list.append(float(x))
                                except (ValueError, TypeError):
                                    continue
                            new_parameters_list = params_list
                        
                        if new_parameters_list:
                            print(f"✓ Parsed {len(new_parameters_list)} parameters from JSON")
                            if "reasoning" in data or "reason" in data or "exploration_reason" in data:
                                reasoning = data.get("reasoning") or data.get("reason") or data.get("exploration_reason")
                                print(f"LLM reasoning: {reasoning}")
                            return new_parameters_list
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
            
            # Fallback: Extract single number (be more careful to avoid extracting multiple numbers)
            # Look for a number that's not part of a longer sequence
            lines = cleaned.split('\n')
            for line in lines[:3]:  # Check first 3 lines only
                # Try to find "param" or "params" followed by a number
                param_match = re.search(r'(?:param|params?)[\s:=]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                if param_match:
                    try:
                        param = float(param_match.group(1))
                        new_parameters_list = [param]
                        print(f"✓ Parsed 1 parameter from pattern matching: {param}")
                        return new_parameters_list
                    except ValueError:
                        pass
            
            # Last resort: find first number in response
            number_match = re.search(r'-?\d+\.?\d+', cleaned)
            if number_match:
                try:
                    param = float(number_match.group())
                    new_parameters_list = [param]
                    print(f"✓ Parsed 1 parameter from first number: {param}")
                    return new_parameters_list
                except ValueError:
                    pass

        except Exception as e:
            print(f"Error parsing parameters: {e}")
            import traceback
            traceback.print_exc()

        print("✗ Failed to parse any parameters")

        return new_parameters_list

    def llm_update_parameters_num_optim(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):

        """Update parameters using LLM"""
        self.reset_llm_conversation()

        # Build prompt using Jinja2 template
        template_data = dict(self.template_vars)
        template_data.update(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": step_number,
                "rank": rank,
                "optimum": optimum,
                "step_size": search_step_size,
                "actions": actions,
            }
        )

        system_prompt = self.llm_si_template.render(template_data)

        self.add_llm_conversation(system_prompt, role="user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )

class SimpleLLMOptimizer:
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):

        # Define Jinja2 template for system prompt
        self.llm_si_template = Template("""
        You are good global optimizer, helping me find the global maximum of a mathematical function f(params).
I will give you the function evaluation and the current iteration number at each step.
Your goal is to propose input values that efficiently lead us to the global maximum within a limited number of iterations ({{ MAX_ITERS }}).

# Regarding the parameters **params**:
**params** is an array of {{ rank }} float numbers.
**params** values are in the range of [{{ "%.1f"|format(param_min) }}, {{ "%.1f"|format(param_max) }}] with 1 decimal place.

# Here's how we'll interact:
1. I will first provide MAX_STEPS ({{ MAX_ITERS }}) along with a few training examples.
2. You will provide your response in the following exact format:
    * Line 1: a new input 'params: ', aiming to maximize the function's value f(params).
    Please propose params values in the range of [{{ "%.1f"|format(param_min) }}, {{ "%.1f"|format(param_max) }}], with 1 decimal place.
    * Line 2: detailed explanation of why you chose that input.
3. I will then provide the function's value f(params) at that point, and the current iteration.
4. We will repeat steps 2-3 until we reach the maximum number of iterations.

# Remember:
1. **DO NOT PROPOSE PREVIOUSLY SEEN PARAMS**
2. **Balance Exploitation and Exploration:**  Early on, explore broadly. As iterations increase, focus more on promising regions.
3. Search both positive and negative values. **DURING EXPLORATION, USE SEARCH STEP SIZE OF {{ step_size }}**.
4. **Be adaptable:**  Your approach might need to change based on the function's behavior and the remaining iterations. If you think you are stuck in a local minima or making small increments for too long, try more exploratory values and then eventually exploit new values based on your understanding of the function.


Next, you will see examples of params and f(params) pairs.
{{ episode_reward_buffer_string }}

Now you are at iteration {{step_number}} out of {{ MAX_ITERS }}. Please provide the results in the indicated format. Do not provide any additional texts.
        """)

        # Initialize LLM brain
        self.brain = LLMBrain(
            llm_si_template=self.llm_si_template,
            llm_model_name=model_name
        )

        # Initialize episode reward buffer
        self.episode_reward_buffer = EpisodeRewardBufferNoBias(max_size=200)

    def plot_1d_animation(self, data_trace, x_range, title, filename, fps=2):
        """Plot 1D function trace"""
        x_values = np.linspace(x_range[0], x_range[1], 400)
        y_values = [self.current_objective_fn([x]) for x in x_values]

        fig, ax = plt.subplots(figsize=(8,6), dpi=120)

        def update(frame):
            ax.clear()

            ax.plot(x_values, y_values, color='black', label='f(params)', zorder=1, lw=2)

            current_trace = data_trace[:frame+1]
            colors = plt.cm.viridis(np.linspace(0, 1, len(current_trace)))

            for step, pt in enumerate(current_trace):
                ax.scatter(pt[0], pt[1], s=130, color=colors[step], marker='o', zorder=2+step)
            
            ax.set_title(f'{title} - Step {frame+1}/{len(data_trace)}')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Function Value f(params)')
            ax.legend()

            norm = mcolors.Normalize(vmin=0, vmax=len(data_trace))
            sm = cm.ScalarMappable(cmap='plasma', norm=norm)
            sm.set_array([])

        anim = FuncAnimation(fig, update, frames=len(data_trace), interval=1000/fps, repeat=True)
        anim.save(filename, writer=PillowWriter(fps=fps))
        plt.close(fig)
    
    def optimize_1(self, objective_fn, param_dim, max_iterations, param_range=(-1.0, 1.0), search_step_size=0.1, test_prefix=""):
        """
        Optimize parameters.
        
        Args:
            objective_fn: Objective function to maximize
            max_iterations: Maximum number of iterations
            param_range: Parameter value range
            search_step_size: Step size for parameter search
            test_prefix: Prefix for wandb logging (e.g., "test1_")
        """
        print(f"\n{'='*60}")
        print(f"Starting Optimization")
        print(f"{'='*60}")
        print(f"Max iterations: {max_iterations}")
        print(f"Parameter range: {param_range}")
        print(f"Search step size: {search_step_size}")
        print(f"{'='*60}\n")

        # Set test prefix for wandb logging
        if test_prefix:
            set_test_prefix(test_prefix)

        # Template variables
        self.template_vars = {
            "MAX_ITERS": max_iterations,
            "rank": param_dim,
            "param_min": param_range[0],
            "param_max": param_range[1],
            "optimum": 100.0,
            "step_size": search_step_size,
        }
        self.brain.template_vars = self.template_vars

        data_trace = []
        self.current_objective_fn = objective_fn
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # 1. Propose parameters
            new_parameters_list, conversation_log, api_time = self.brain.llm_update_parameters_num_optim(
                episode_reward_buffer=self.episode_reward_buffer,
                parse_parameters=self.brain.parse_parameters,
                step_number=iteration + 1,
                rank=param_dim,
                optimum=self.template_vars['optimum'],
                search_step_size=self.template_vars['step_size'],
                actions=None,
            )

            print(f"Proposed params: {new_parameters_list}")
            print(f"Reasoning time: {api_time:.2f}s")
            
            # 2. Evaluate with objective function
            reward = objective_fn(new_parameters_list)
            print(f"Reward: {reward:.4f}")
            
            # 3. Add to buffer
            self.episode_reward_buffer.add(new_parameters_list, reward)

            # Update data trace for plotting
            data_trace.append((new_parameters_list[0], reward))
            
            # 4. Display best value
            best_reward = max([r for _, r in self.episode_reward_buffer.buffer])
            print(f"Best so far: {best_reward:.4f}")

            log_iteration(
                iteration=iteration + 1,
                proposed_parameters=new_parameters_list,
                reward=reward,
                reasoning_time=api_time,
                reasoning=conversation_log
            )
        
        # Final result
        best_params, best_reward = max(self.episode_reward_buffer.buffer, key=lambda x: x[1])
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best params: {best_params}")
        print(f"Best reward: {best_reward:.4f}")
        print(f"{'='*60}\n")

        # Plot final trace
        if data_trace:
            gif_filename = f"{test_prefix}optimization_trace.gif"
            self.plot_1d_animation(
                data_trace=data_trace,
                x_range=param_range,
                title=f'{test_prefix}Optimization',
                filename=gif_filename,
                fps=2
            )

            wandb.log({
                    f"{test_prefix}optimization_animation": wandb.Video(gif_filename, fps=2, format="gif")
                })
        
        return best_params, best_reward
    
    def optimize_2(self, objective_fn, param_dim, max_iterations, param_range=(-1.0, 1.0), search_step_size=0.1, test_prefix=""):
        """
        Optimize parameters.
        
        Args:
            objective_fn: Objective function to maximize
            max_iterations: Maximum number of iterations
            param_range: Parameter value range
            search_step_size: Step size for parameter search
            test_prefix: Prefix for wandb logging (e.g., "test2_")
        """
        print(f"\n{'='*60}")
        print(f"Starting Optimization")
        print(f"{'='*60}")
        print(f"Max iterations: {max_iterations}")
        print(f"Parameter range: {param_range}")
        print(f"Search step size: {search_step_size}")
        print(f"{'='*60}\n")

        # Set test prefix for wandb logging
        if test_prefix:
            set_test_prefix(test_prefix)

        # Template variables
        self.template_vars = {
            "MAX_ITERS": max_iterations,
            "rank": param_dim,
            "param_min": param_range[0],
            "param_max": param_range[1],
            "optimum": 100.0,
            "step_size": search_step_size,
        }
        self.brain.template_vars = self.template_vars

        data_trace = []
        self.current_objective_fn = objective_fn
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # 1. Propose parameters
            new_parameters_list, conversation_log, api_time = self.brain.llm_update_parameters_num_optim(
                episode_reward_buffer=self.episode_reward_buffer,
                parse_parameters=self.brain.parse_parameters,
                step_number=iteration + 1,
                rank=param_dim,
                optimum=self.template_vars['optimum'],
                search_step_size=self.template_vars['step_size'],
                actions=None,
            )

            print(f"Proposed params: {new_parameters_list}")
            print(f"Reasoning time: {api_time:.2f}s")
            
            # 2. Evaluate with objective function
            reward = objective_fn(new_parameters_list)
            print(f"Reward: {reward:.4f}")
            
            # 3. Add to buffer
            self.episode_reward_buffer.add(new_parameters_list, reward)

            # Update data trace for plotting
            data_trace.append((new_parameters_list[0], reward))
            
            # 4. Display best value
            best_reward = max([r for _, r in self.episode_reward_buffer.buffer])
            print(f"Best so far: {best_reward:.4f}")

            log_iteration(
                iteration=iteration + 1,
                proposed_parameters=new_parameters_list,
                reward=reward,
                reasoning_time=api_time,
                reasoning=conversation_log
            )
        
        # Final result
        best_params, best_reward = max(self.episode_reward_buffer.buffer, key=lambda x: x[1])
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best params: {best_params}")
        print(f"Best reward: {best_reward:.4f}")
        print(f"{'='*60}\n")

        # Plot final trace
        if data_trace:
            gif_filename = f"{test_prefix}optimization_trace.gif"
            self.plot_1d_animation(
                data_trace=data_trace,
                x_range=param_range,
                title=f'{test_prefix}Optimization',
                filename=gif_filename,
                fps=2
            )

            wandb.log({
                    f"{test_prefix}optimization_animation": wandb.Video(gif_filename, fps=2, format="gif")
                })
        
        return best_params, best_reward


# ===================================================================
# Objective Functions for Testing
# ===================================================================

def test_quadratic_function(params: np.ndarray) -> float:
    """
    Test 1: Simple Quadratic Function
    f(params) = -sum((params - target)^2)
    Target: params = [1.0]
    Maximum reward = 100.0
    """
    params = np.array(params)
    target = 1.0
    error = np.sum((params - target) ** 2)
    reward = 100.0 - error * 10.0
    return reward


def test_nonlinear_function(params: np.ndarray) -> float:
    """
    Test 3: Nonlinear Complex Function
    This function has multiple local optima
    """

    params = params[0]

    score = -(params**2) + 5 * np.cos(2*params)

    # Shift reward to be around 100 at optimum
    reward = score + 95.0
    
    return reward


def main():
    # Start single wandb run for both tests
    init_wandb(
        project_name="LLM_Optimization Test 3", 
        run_name="Parameter Optimization Qwen Changed Decimal", 
        config={
            "num_tests": 2,
            "test1_name": "Quadratic Function",
            "test1_param_dim": 1,
            "test1_max_iterations": 50,
            "test1_param_range": (-3.0, 3.0),
            "test2_name": "Nonlinear Function",
            "test2_param_dim": 1,
            "test2_max_iterations": 50,
            "test2_param_range": (-4.0, 4.0),
        }
    )
    
    # Initialize optimizer
    optimizer = SimpleLLMOptimizer(
        model_name="Qwen/Qwen2.5-14B-Instruct"
    )
    
    # ===== Test 1: Simple Quadratic Function =====
    print("\n" + "#"*60)
    print("# TEST 1: Simple Quadratic Function")
    print("#"*60)
    print("Target: params = [1.0]")
    print("Maximum reward: 100.0")
    
    best_params_1, best_reward_1 = optimizer.optimize_1(
        objective_fn=test_quadratic_function,
        param_dim=1,
        max_iterations=50,
        param_range=(-3.0, 3.0),
        test_prefix="test1_"
    )
    
    #input("\nPress Enter to continue to Test 2...")
    
    # Reset buffer for Test 2
    optimizer.episode_reward_buffer = EpisodeRewardBufferNoBias(max_size=200)
    
    # ===== Test 2: Nonlinear Function =====
    print("\n" + "#"*60)
    print("# TEST 2: Nonlinear Function (Rastrigin-like)")
    print("#"*60)
    print("This has multiple local optima")
    print("Testing LLM's global search ability")
    
    best_params_2, best_reward_2 = optimizer.optimize_2(
        objective_fn=test_nonlinear_function,
        param_dim=1,
        max_iterations=50,
        param_range=(-4.0, 4.0),
        test_prefix="test2_"
    )
    
    # ===== Final Summary =====
    print("\n" + "="*60)
    print("SUMMARY OF ALL TESTS")
    print("="*60)
    print(f"Test 1 (Quadratic):  Best reward = {best_reward_1:.4f} / 100.0")
    print(f"Test 2 (Nonlinear):  Best reward = {best_reward_2:.4f} / 100.0")
    print("="*60)
    
    # Finish wandb run after both tests
    finish_wandb()


if __name__ == "__main__":
    main()