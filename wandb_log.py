import wandb

# Global variable to track current test prefix
current_test_prefix = ""

def init_wandb(project_name, run_name, config):
    """Initialize wandb run"""
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=False  # Don't reinitialize if already running
    )
    print(f"Wandb initialized: {project_name}/{run_name}")


def set_test_prefix(prefix):
    """Set the prefix for the current test"""
    global current_test_prefix
    current_test_prefix = prefix
    print(f"Set test prefix to: {prefix}")


def log_iteration(iteration, proposed_parameters, reward, reasoning_time, reasoning):
    """Log iteration data with test prefix"""
    global current_test_prefix
    
    # Create log dictionary with prefix
    log_dict = {
        f"{current_test_prefix}iteration": iteration,
        f"{current_test_prefix}reward": reward,
        f"{current_test_prefix}reasoning_time": reasoning_time,
    }
    
    # Log parameters
    for i, param in enumerate(proposed_parameters):
        log_dict[f"{current_test_prefix}param_{i}"] = param
    
    wandb.log(log_dict)
    
    # Optionally log reasoning as text
    if reasoning and iteration % 5 == 0:  # Log reasoning every 5 iterations to avoid clutter
        wandb.log({f"{current_test_prefix}reasoning": wandb.Html(f"<pre>{reasoning}</pre>")})


def finish_wandb():
    """Finish wandb run"""
    wandb.finish()
    print("Wandb run finished")