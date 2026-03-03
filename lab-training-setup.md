This is the consolidated `.md` code for your final takeaway guide. I have streamlined the instructions, removed redundancies, and formatted it for maximum readability when viewed in a Markdown previewer or GitHub/GitLab.

```markdown
# 🚀 CW1 Lab Machine & GPU Cluster Guide

**User**: as9422  
**Project**: Deep Learning Coursework 1 (Food-101)  
**Storage Path**: `/vol/bitbucket/as9422/dl_cw_1/`

---

## 📍 Quick Strategy: Where to Train?

| Phase | Task | Recommended Environment |
| :--- | :--- | :--- |
| **Part 1** | Layer Implementation | **Local Mac** or **Lab PC** (CPU is fine) |
| **Part 2** | Food-101 Training | **DoC GPU Cluster** (SLURM) or **GPU Dojo** |
| **Debug** | Environment Issues | **Lab Workstation** (Direct GPU access via SSH) |

---

## 🛠️ Infrastructure Setup (DoC Cluster)

### 1. SSH Access (From Mac)
```bash
# Off-campus (requires DoC private key)
ssh -i ~/.ssh/doclab_ecdsa -J shell3 as9422@gpucluster2.doc.ic.ac.uk

# On-campus WiFi
ssh as9422@gpucluster2.doc.ic.ac.uk

```

### 2. Repository & Venv Setup

> **Important**: Always use `/vol/bitbucket/` to avoid "Disk Quota Exceeded" errors in your home directory.

```bash
cd /vol/bitbucket/as9422/
git clone [https://gitlab.doc.ic.ac.uk/lab2526_spring/70010_DL_CW_1_as9422.git](https://gitlab.doc.ic.ac.uk/lab2526_spring/70010_DL_CW_1_as9422.git) dl_cw_1
cd dl_cw_1

# Create environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install requirements (Adjust for Python 3.12 if needed)
pip install torch torchvision
pip install otter-grader==6.0.4 numpy==1.26.4 matplotlib==3.10.0 tqdm==4.67.1 scikit-learn==1.6.0 seaborn==0.13.2 jupyter

```

---

## 🖥️ Option A: Interactive Training (Lab Workstations)

Best for real-time debugging and visualization.

1. **Pick a machine**: Find an available GPU node at [CSG Workstations](https://www.imperial.ac.uk/computing/people/csg/facilities/lab/workstations/).
2. **Tunnel from Mac**:
```bash
ssh -L 8888:localhost:8888 -A -J as9422@shell3.doc.ic.ac.uk as9422@{LAB_MACHINE}.doc.ic.ac.uk

```


3. **Start Jupyter (on Lab Machine)**:
```bash
source /vol/bitbucket/as9422/dl_cw_1/venv/bin/activate
jupyter notebook --no-browser --port=8888

```


4. **Access**: Open `http://localhost:8888` on your Mac browser.

---

## ⚖️ Option B: Batch Training (SLURM Cluster)

Best for long-running training sessions (Food-101).

### 1. Interactive GPU Allocation

```bash
# Request a T4 or A16 GPU for 4 hours
salloc --gres=gpu:1 --partition=a16 --time=04:00:00

```

### 2. Batch Script (`train_food101.sh`)

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/as9422/dl_cw_1/logs/job_%j.out

source /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/as9422/dl_cw_1/venv/bin/activate
cd /vol/bitbucket/as9422/dl_cw_1

# Run the notebook and save results in place
jupyter nbconvert --to notebook --execute --inplace dl_cw_1.ipynb

```

**Submit with**: `sbatch train_food101.sh`

---

---

## 📝 Notebook Configuration (Cell 3)

### On Lab/Cluster:

Set `WORKING_ENV = 'LABS'` and `USERNAME = 'as9422'`. Cell 3 will automatically detect `/vol/bitbucket/as9422/dl_cw_1/`.

### On Local Mac:

Cell 3 will fail to find `/vol/`. **Immediately after Cell 3**, add this override:

```python
# LOCAL OVERRIDE
import os
content_path = os.path.abspath('.') 
print(f"Local Path Set: {content_path}")

```

---

## ⚠️ Troubleshooting Checklist

* **Quota Full?** Run `quota -v`. Delete `~/.cache` or move large files to `/vol/bitbucket/`.
* **Module Not Found?** Ensure you ran `python -m ipykernel install --user --name cw1-venv` and selected that kernel in the Jupyter UI.
* **CUDA Errors?** Ensure you are on a GPU node (run `nvidia-smi`). If it returns an error, you are likely still on the login/head node.
* **Port 8888 Busy?** Change the port in your SSH command (`-L 8889:localhost:8889`) and Jupyter command (`--port=8889`).

---

*Updated for 2025/26 Lab Cycle*

```

Would you like me to generate the specific `sbatch` submission script as a separate file for you?

```


The t4 gpuvm[14,17-19,24],kingfisher seem to be idle right now. use that. if you have questions. ask them. i've got more information. 