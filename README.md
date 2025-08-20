# SeizAware 🧠⚡  
**EEG-based Epileptic Seizure Prediction & Decision Support Tool**

SeizAware is a machine learning + deep learning project aimed at detecting epileptic seizures from EEG (electroencephalogram) signals. It provides a **clinician-facing decision-support system** that prioritizes **high sensitivity (recall)** to ensure seizures are not missed.  

The project demonstrates a full technical workflow: from **data preprocessing → model building → evaluation → deployment as an interactive Streamlit app**.

---

## 🔑 Features
- **Datasets Supported**  
  - [UCI Epileptic Seizure Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition) (178 features/sample, preprocessed).  
  - [Bonn University EEG Dataset](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3) (raw EEG, widely cited).  

- **Data Preprocessing**  
  - Band-pass filtering (0.5–45 Hz) & notch filtering (50 Hz).  
  - Sliding-window segmentation (1–2 seconds).  
  - Normalization & optional artifact removal.  
  - Feature extraction (statistical, frequency, entropy) for classical ML.

- **Models Implemented**  
  - Classical ML: Logistic Regression, Random Forest, SVM, XGBoost.  
  - Deep Learning: 1D CNN, CNN+LSTM/GRU for temporal patterns.  
  - **Evaluation:** LOSO / subject-wise splits to avoid data leakage.

- **Evaluation Metrics**  
  - Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC.  
  - Special focus on **Sensitivity** to minimize missed seizures.  
  - False alarms/hour analysis.  

- **Interactive Prototype (Streamlit App)**  
  - Upload EEG data → model predicts seizure likelihood.  
  - Visualize EEG segments with high-risk regions highlighted.  
  - Displays seizure probability score.  
