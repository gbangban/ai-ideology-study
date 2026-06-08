import sys
sys.modules["llm_blender"] = type(sys)("llm_blender")
from trl import GRPOTrainer
print("OK")
