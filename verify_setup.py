"""
Verification script to check if all required files and dependencies are in place
for the puf-02 FWL sentiment analysis implementation.
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {filepath}")
    return exists

def check_module(module_name):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"[OK] Module '{module_name}' is installed")
        return True
    except ImportError:
        print(f"[MISSING] Module '{module_name}' is NOT installed")
        return False

def main():
    print("="*70)
    print("PUF-02 FWL Setup Verification")
    print("="*70)

    all_ok = True

    # Check Python files
    print("\n1. Checking Python scripts...")
    all_ok &= check_file("main_bilstm_fwl.py", "Main script")
    all_ok &= check_file("NegationHandlingBaseline.py", "FWL negation module")

    # Check documentation
    print("\n2. Checking documentation...")
    all_ok &= check_file("README.md", "README documentation")

    # Check dataset files
    print("\n3. Checking dataset files...")
    all_ok &= check_file("data/dataset/partitioned/train-prdct-id.csv", "Training data")
    all_ok &= check_file("data/dataset/partitioned/val-prdct-id.csv", "Validation data")
    all_ok &= check_file("data/dataset/partitioned/test-prdct-id.csv", "Test data")

    # Check directories
    print("\n4. Checking output directories...")
    all_ok &= check_file("models/", "Models directory")
    all_ok &= check_file("results/", "Results directory")

    # Check external dependencies
    print("\n5. Checking external dependencies...")
    all_ok &= check_file("../resources/taggers/example-universal-pos/best-model.pt", "POS tagger model")

    # Check Python modules
    print("\n6. Checking Python module dependencies...")
    required_modules = [
        'numpy',
        'pandas',
        'matplotlib',
        'tensorflow',
        'keras',
        'gensim',
        'sklearn',
        'tqdm',
        'flair'
    ]

    for module in required_modules:
        all_ok &= check_module(module)

    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("SUCCESS: All checks passed! You're ready to run the experiment.")
        print("\nTo start training, run:")
        print("  python main_bilstm_fwl.py")
    else:
        print("FAILED: Some checks failed. Please fix the issues above before running.")
        print("\nTo install missing Python packages, run:")
        print("  pip install numpy pandas matplotlib tensorflow gensim scikit-learn tqdm flair")
    print("="*70)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
