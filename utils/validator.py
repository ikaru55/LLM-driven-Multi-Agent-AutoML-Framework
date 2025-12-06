import torch
import sys
import os
import importlib.util


def check_syntax(file_path):
    """Return True if the Python file compiles without syntax errors."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        compile(source, file_path, "exec")
        return True, "Syntax OK"
    except Exception as e:
        return False, f"Syntax Error: {str(e)}"


def check_model_constraints(model_path, max_params=20000000):
    """Validate get_model output against param budget and forward pass."""
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"Import Error: {str(e)}"

    try:
        if not hasattr(module, "get_model"):
            return False, "Error: get_model() function not found."

        model = module.get_model()
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > max_params:
            return False, f"Parameter Count Exceeded: {total_params} > {max_params}"
        dummy_input = torch.randn(1, 3, 112, 112)
        try:
            _ = model(dummy_input)
        except Exception as e:
            return False, f"Forward Pass Error: {str(e)}"

        return True, f"Validation Passed. Params: {total_params}"

    except Exception as e:
        return False, f"Runtime Error during validation: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validator.py <model_path>")
        sys.exit(1)

    path = sys.argv[1]
    success, msg = check_syntax(path)
    if not success:
        print(msg)
        sys.exit(1)

    success, msg = check_model_constraints(path)
    print(msg)
    if not success:
        sys.exit(1)
