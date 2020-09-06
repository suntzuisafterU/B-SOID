# Include at top of Notebook header to inject BSOID project path into sys.path
import os; import sys
bsoid_PROJECT_path = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
if bsoid_PROJECT_path not in sys.path:
    sys.path.insert(0, bsoid_PROJECT_path)
