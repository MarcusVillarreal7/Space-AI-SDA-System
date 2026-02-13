# 📓 Jupyter Notebook Setup Guide

**Problem**: Jupyter notebook can't find project dependencies (loguru, etc.)

**Cause**: Jupyter is not using the project's virtual environment

---

## Solution: Install Jupyter in Virtual Environment

### Step 1: Activate Virtual Environment
```bash
cd /home/marcus/Cursor-Projects/space-ai
source venv/bin/activate
```

### Step 2: Install Jupyter in Virtual Environment
```bash
pip install jupyter ipykernel
```

### Step 3: Register Virtual Environment as Jupyter Kernel
```bash
python -m ipykernel install --user --name=space-ai --display-name="Python (space-ai)"
```

### Step 4: Launch Jupyter from Virtual Environment
```bash
# Make sure you're in the project root
cd /home/marcus/Cursor-Projects/space-ai

# Launch Jupyter
jupyter notebook
```

### Step 5: Select Correct Kernel in Notebook
1. Open `notebooks/01_data_exploration.ipynb`
2. Click **Kernel** → **Change Kernel** → **Python (space-ai)**
3. Run the cells

---

## Alternative: Quick Fix for Current Session

If you already have Jupyter running, you can install packages in the current kernel:

```python
# Run this in the first cell of your notebook
import sys
!{sys.executable} -m pip install loguru skyfield sgp4 pandas numpy matplotlib seaborn tqdm pyarrow
```

Then restart the kernel and run your notebook.

---

## Verify Setup

Run this in a notebook cell to verify:

```python
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Try importing key packages
try:
    import loguru
    import skyfield
    import pandas
    print("✅ All packages available!")
except ImportError as e:
    print(f"❌ Missing package: {e}")
```

---

## Recommended Workflow

### Option 1: Jupyter Lab (Recommended)
```bash
# Install JupyterLab in virtual environment
pip install jupyterlab

# Launch JupyterLab
jupyter lab

# Navigate to notebooks/01_data_exploration.ipynb
```

### Option 2: VS Code with Jupyter Extension
1. Install "Jupyter" extension in VS Code
2. Open `notebooks/01_data_exploration.ipynb`
3. Click "Select Kernel" in top right
4. Choose "Python Environments..." → Select `venv/bin/python`

### Option 3: Command Line Jupyter
```bash
# From project root with venv activated
cd /home/marcus/Cursor-Projects/space-ai
source venv/bin/activate
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Troubleshooting

### Issue: Kernel not showing up
**Solution**:
```bash
# List available kernels
jupyter kernelspec list

# If space-ai not listed, register it again
python -m ipykernel install --user --name=space-ai --display-name="Python (space-ai)"
```

### Issue: Still getting ModuleNotFoundError
**Solution**:
```bash
# Verify you're in the virtual environment
which python
# Should show: /home/marcus/Cursor-Projects/space-ai/venv/bin/python

# Verify packages are installed
pip list | grep -E "(loguru|skyfield|pandas)"

# If not, install requirements
pip install -r requirements.txt
```

### Issue: Can't find src module
**Solution**: Add this to the first cell of your notebook:
```python
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

print(f"Added to path: {project_root}")
```

---

## Complete Setup Script

Save this as `setup_jupyter.sh`:

```bash
#!/bin/bash
# setup_jupyter.sh - Set up Jupyter for Space AI project

set -e

echo "🚀 Setting up Jupyter for Space AI project"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Install Jupyter packages
echo "Installing Jupyter packages..."
pip install jupyter jupyterlab ipykernel

# Register kernel
echo "Registering kernel..."
python -m ipykernel install --user --name=space-ai --display-name="Python (space-ai)"

# Verify installation
echo ""
echo "✅ Setup complete!"
echo ""
echo "Available kernels:"
jupyter kernelspec list

echo ""
echo "To launch Jupyter:"
echo "  jupyter notebook"
echo "  # or"
echo "  jupyter lab"
echo ""
echo "Then select kernel: Python (space-ai)"
```

**Usage**:
```bash
chmod +x setup_jupyter.sh
./setup_jupyter.sh
```

---

## Quick Start (TL;DR)

```bash
# 1. Activate virtual environment
cd /home/marcus/Cursor-Projects/space-ai
source venv/bin/activate

# 2. Install Jupyter
pip install jupyter ipykernel

# 3. Register kernel
python -m ipykernel install --user --name=space-ai --display-name="Python (space-ai)"

# 4. Launch Jupyter
jupyter notebook

# 5. In notebook: Kernel → Change Kernel → Python (space-ai)
```

---

**Last Updated**: 2026-02-04  
**Status**: Tested and verified
