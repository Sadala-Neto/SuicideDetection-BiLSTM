# 📂 `data/` Folder

This folder should contain the data files required for training and evaluating the model.

⚠️ **Attention:**  
For **licensing, privacy, and ethical** reasons, **the original data is not included** in this repository.  
The data used is third-party and must be downloaded separately.

---

## 📦 Dataset used

- **Name**: Suicide and Depression Detection
- **Source**: Kaggle
- **Download link**: [https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **Author(s)**: Nikhileswar Komati et al.
- **Format**: CSV
- **Classes**:
  - `suicide`: posts with indications of suicidal ideation
  - `non-suicide`: posts without indications of suicidal ideation
- **Total samples**: 232,074 posts (balanced across two classes)

---

## 🛠️ Usage instructions

1. **Download the dataset**  
   Go to the Kaggle link above and download the ZIP file.

2. **Extract the contents**  
   Extract the downloaded file to obtain the `.csv` files.

3. **Organize the data**  
   Place the `.csv` files inside the `data/` folder of this repository.

4. **Verify file paths**  
   Ensure the CSV filename matches what is configured in the code (default: `Suicide_Detection.csv`).  
   If the file has a different name or path, update it in the code.

---

## 📜 Ethical note

The dataset contains **sensitive posts** about suicidal ideation.  
Use it **only for academic or research purposes**.  
Handling of this data must follow ethical guidelines and, whenever possible, be supervised by mental health professionals.

---

## 📄 Licensing

Refer to the Kaggle page for terms of use and licensing.  

This repository **does not distribute** the dataset, it only provides instructions on how to obtain it.
