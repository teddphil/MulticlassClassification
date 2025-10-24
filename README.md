# **Generalized Multi-Class Classification and Visualization Pipeline**

This project is a refactored, modular pipeline designed for end-to-end multi-class classification and statistical visualization of tabular data. It separates application-specific configurations from generalized quantitative tools, making it highly reusable for various analytical tasks, such as those encountered in data science and quantitative research.

## **🚀 Key Features**

* **Modular Architecture:** Clear separation between logic (ml\_toolkit.py), configuration (pet\_config.py, style\_config.py), and execution (main.py).  
* **Multi-Class ML:** Performs per-feature and multi-feature classification using **LDA**, **Logistic Regression**, and other models, validated via **Leave-One-Out Cross-Validation (LOOCV)**.  
* **Dimensionality Reduction:** Automated generation of **PCA** and **t-SNE** plots for high-dimensional feature spaces, helping to visualize class separation.  
* **Statistical Visualization:** Generates publication-quality **Violin Plots** with Mann-Whitney U statistical annotations for pairwise group comparisons.  
* **Code Quality:** All variables adhere to **snake\_case** conventions, and domain-specific terms (like "biomarker") have been generalized to "features."

## **📁 Project Structure**

| File | Role | Description |
| :---- | :---- | :---- |
| main.py | **Execution Engine** | Loads configuration and orchestrates the entire pipeline. |
| toolkit.py | **Quantitative Toolkit** | Contains all reusable functions for ML (AUC, LOOCV) and plotting. This file is entirely data-agnostic. |
| style.py | **Styling** | Defines all visual parameters (fonts, colors, figure sizes) for consistent output. |

## **⚙️ Usage**

1. **Configure Data:** Update the DATA\_PATH and the ANALYSIS\_CONFIGS (specifically the grouping\_map and tumour\_groups) within pet\_config.py to match your input data and desired analytical comparisons.  
2. **Adjust Parameters:** Modify settings in GLOBAL\_CONFIG\['plotting\_parameters'\] to control feature selection (e.g., max\_num\_features), cross-validation, and plot generation.  
3. **Run Pipeline:** Execute the main script:  
   python main.py

Results (plots, CSVs, and model performance metrics) will be saved in the configured Results directory.
