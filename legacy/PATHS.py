# PATHS.py - Having all the Paths in one place for easier organizing

# Imports
import os

# To standardise the parent directory
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(THIS_FILE)

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# Output folder's subfolders
MODIFIED_ADVBENCH_PATH = os.path.join(OUTPUT_DIR, 'modified_advbench')
XSTEST_PATH = os.path.join(OUTPUT_DIR, 'xstest')

# Subfolders under Modified Advbench
MA_MISTRAL_7B = os.path.join(MODIFIED_ADVBENCH_PATH, 'mistral_7b_instruct')
MA_MISTRAL_12B = os.path.join(MODIFIED_ADVBENCH_PATH, 'mistral_12b_instruct')
MA_QWEN_7B = os.path.join(MODIFIED_ADVBENCH_PATH, 'qwen_7b_instruct')
MA_QWEN_14B = os.path.join(MODIFIED_ADVBENCH_PATH, 'qwen_14b_instruct')

# Subfolders under XSTest
XST_MISTRAL_7B = os.path.join(XSTEST_PATH, 'mistral_7b_instruct')
XST_MISTRAL_12B = os.path.join(XSTEST_PATH, 'mistral_12b_instruct')
XST_QWEN_7B = os.path.join(XSTEST_PATH, 'qwen_7b_instruct')
XST_QWEN_14B = os.path.join(XSTEST_PATH, 'qwen_14b_instruct')