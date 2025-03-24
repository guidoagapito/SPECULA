import os
import ast
import yaml
import sys

def extract_class_info(file_path):
    """Extracts class name, __init__ method parameters, default values, and types from a Python file."""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    class_data = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            init_params = {}
            param_comments = {}

            # Find __init__ method
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef) and sub_node.name == "__init__":
                    total_params = len(sub_node.args.args) - 1  # Exclude 'self'
                    num_defaults = len(sub_node.args.defaults)

                    # Extract type hints from function annotations
                    annotations = {arg.arg: ast.unparse(arg.annotation) if arg.annotation else None for arg in sub_node.args.args[1:]}  # Skip 'self'

                    for i, arg in enumerate(sub_node.args.args[1:]):  # Skip 'self'
                        param_name = arg.arg
                        param_type = annotations.get(param_name, None)
                        is_optional = i >= (total_params - num_defaults)
                        
                        default_value = "value"
                        if is_optional:
                            default_index = i - (total_params - num_defaults)
                            default_node = sub_node.args.defaults[default_index]
                            
                            try:
                                default_value = ast.literal_eval(default_node)
                            except (ValueError, TypeError, AttributeError):
                                if isinstance(default_node, ast.Name):
                                    default_value = default_node.id  # Handle names like `np`
                                else:
                                    default_value = "value"

                        # Construct comment with type and optional status
                        comment = "Required" if not is_optional else f"Optional (default={default_value})"
                        if param_type:
                            comment += f", type: {param_type}"

                        init_params[param_name] = default_value
                        param_comments[param_name] = comment
            
            class_data.append((class_name, init_params, param_comments))
    
    return class_data

def generate_yaml(class_name, params, comments, output_folder):
    """Generates a YAML file with class information and comments for each parameter."""
    yaml_path = os.path.join(output_folder, f"{class_name}.yml")
    
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(f"example_{class_name}:\n")
        for param, value in params.items():
            yaml_file.write(f"  {param}: {value}  # {comments[param]}\n")
    
    print(f"Generated YAML: {yaml_path}")

def process_python_files(input_folder, output_folder):
    """Processes all Python files in a directory and generates YAML files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".py"):
            file_path = os.path.join(input_folder, file_name)
            classes = extract_class_info(file_path)
            
            for class_name, params, comments in classes:
                generate_yaml(class_name, params, comments, output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_classes.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    process_python_files(input_folder, output_folder)

