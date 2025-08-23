# Domain-Adaptation-with-Fuzzy-Transfer-Learning-and-DANN

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Domain Adaptation](https://img.shields.io/badge/Domain-Adaptation-lightgrey?style=flat-square)
![Transfer Learning](https://img.shields.io/badge/Transfer-Learning-green?style=flat-square)
![Adversarial Networks](https://img.shields.io/badge/Adversarial-Networks-red?style=flat-square)



This project explores transfer learning and domain adaptation for robust machine learning across datasets with distribution shifts.
It combines:

- Fuzzy Transfer Learning for uncertainty-aware feature adaptation

- DANN (Domain-Adversarial Neural Networks) for domain-invariant representation learning.

---
## ✨ Highlights

- Implements fuzzy logic–based transfer learning for handling noisy and uncertain domain boundaries.

- Uses adversarial training via DANN to align source and target distributions.

- Tested on cross-domain classification tasks.

- Demonstrates improvements in generalization across domains.
---
## 📂 Project Structure

domain-adaptation-fuzzy-dann/
├── LICENSE
├── README.md
├── Fuzzy_transfer_learning_Sarah_Altalhi.ipynb # Fuzzy transfer learning experiments
├── DANN_Sarah_Altalhi_(1).ipynb # Domain-Adversarial NN experiments
│
├── data/ # Dataset folder (ignored by git)
├── experiments/ # Logs, metrics, and visualizations
├── models/ # Saved models and checkpoints
├── scripts/ # Helper scripts for training/evaluation
└── src/ # Core implementation (data loaders, training loops, utils)

---
## ⚙️ Setup
### 1. Clone the repository
git clone https://github.com/Sarah-Altalhi/domain-adaptation-fuzzy-dann.git
cd domain-adaptation-fuzzy-dann

### 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate   # On Linux/Mac

### 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

---
##🚀 Running the Notebooks

Open the notebooks in Jupyter or VSCode:

### Fuzzy_transfer_learning_Sarah_Altalhi.ipynb:
Implements fuzzy logic–based transfer learning with uncertainty-aware adaptation.

### DANN_Sarah_Altalhi_(1).ipynb:
Implements domain-adversarial training with gradient reversal for robust domain adaptation.

---
## 📊 Results

The experiments showed promising results for **cross-domain generalization**:

| Method                          | Accuracy (Target Domain) | Notes                         |
|---------------------------------|---------------------------|-------------------------------|
| Baseline (Source Only)          | 0.72                      | No adaptation                 |
| Fuzzy Transfer Learning         | 0.80                      | Better handling of uncertainty|
| DANN (Domain Adversarial)       | 0.83                      | Strong domain-invariant features |
| Combined Approach (Fuzzy + DANN)| **0.86**                  | Best performance              |


---
### 📚 References

Ganin et al., 2016. Domain-Adversarial Training of Neural Networks.

Pan & Yang, 2010. A Survey on Transfer Learning.
