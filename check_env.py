import sys
import transformers

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Transformers Path: {transformers.__file__}")

# Check TrainingArguments signature (optional, might fail if class itself is broken)
try:
    from transformers import TrainingArguments
    import inspect
    sig = inspect.signature(TrainingArguments)
    print("\nTrainingArguments parameters:")
    for param in sig.parameters:
        print(f"- {param}")
except Exception as e:
    print(f"\nCould not inspect TrainingArguments: {e}")

