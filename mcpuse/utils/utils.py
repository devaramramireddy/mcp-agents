import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def remove_echo(user_prompt, output):
    # Avoid echoing user prompt at beginning of model output
    if output.strip().lower().startswith(user_prompt.strip().lower()):
        return output[len(user_prompt):].lstrip(" \n.:")
    return output
