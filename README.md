Anomaly Detection in Network Traffic Using Machine Learning
Overview
This project implements a machine learning-based Intrusion Detection System (IDS) focusing on anomaly detection in network traffic. It leverages feature selection techniques (ANOVA F-test and Recursive Feature Elimination) and a Decision Tree classifier to identify malicious behavior, such as DoS, Probe, R2L, and U2R attacks, on the NSL-KDD dataset. By carefully selecting features, we balance detection accuracy with computational efficiency, making it feasible for near real-time deployment in high-speed network environments.
Key Features
* NSL-KDD Dataset: A refined version of KDD Cup 1999 data, standard for evaluating IDS research.
* Feature Engineering:
   * One-Hot Encoding for categorical variables (e.g., protocol_type, service, flag).
   * Standardization of numeric features to ensure balanced model inputs.
* Feature Selection:
   * ANOVA F-test for univariate ranking.
   * Recursive Feature Elimination (RFE) to distill the feature set down to a minimal and effective subset.
* Classification:
   * A Decision Tree model serves as the baseline, chosen for its interpretability and quick training time.
   * Trains separate binary classifiers (normal vs. each attack category) for focused analysis.
Results
* Achieves high detection rates, notably ~99% accuracy for DoS attacks.
* Maintains strong performance across other categories (Probe, R2L, U2R).
* Reducing features significantly decreases computational overhead, enabling near real-time analysis in large-scale, high-throughput networks.






Requirements
* Python 3.9+ recommended.
* Libraries:
   * scikit-learn for machine learning (>=0.24).
   * pandas, numpy for data manipulation.
   * matplotlib, seaborn for visualization (optional).
Installation & Setup
* Clone the Repository:
bash
Copy code
git clone https://github.com/yourusername/anomaly-detection-ml.git
* cd anomaly-detection-ml


   * Set up a Virtual Environment (optional but recommended):
bash
Copy code
python -m venv venv
   * source venv/bin/activate  # On Windows: venv\Scripts\activate


      * Install Dependencies:
bash
Copy code
pip install -r requirements.txt
      * If requirements.txt is not provided, install manually:
bash
Copy code
pip install numpy pandas scikit-learn matplotlib seaborn


Prepare the Dataset:
         * Download the NSL-KDD dataset files (KDDTrain+_2.csv, KDDTest+_2.csv) from a known source.
         * Place them in the data directory or update the code paths accordingly.


Usage
         * Run the Analysis Script:
bash
Copy code
python run_analysis.py
            1. This script will:
            * Load and preprocess the NSL-KDD data.
            * Perform feature selection (ANOVA + RFE).
            * Train and evaluate the Decision Tree models.
            * Print out accuracy, precision, recall, and F1-scores.
            * Optionally plot confusion matrices and ROC curves if visualization code is included.
            2. Adjusting Parameters:
            * You can tweak the number of features selected, the classifier parameters, or the train/test splits inside the script.
            * For different ML models or hyperparameters, edit the code in model_training.py (if such a file is present).
Project Structure


            * anomaly-detection-ml/
            * │
            * ├─ data/
            * │  ├─ KDDTrain+_2.csv
            * │  ├─ KDDTest+_2.csv
            * │
            * ├─ src/
            * │  ├─ Final.ipynb
            * └─ README.md














Acknowledgments
This work references open-source datasets (NSL-KDD) and relies on standard Python ML tools. We also acknowledge online resources and community tutorials that helped refine certain implementation details.
Future Work
            * Integrating other ML methods (Random Forests, XGBoost, or deep learning) for potentially better performance.
            * Implementing unsupervised or semi-supervised techniques (Isolation Forest, One-Class SVM) to detect zero-day attacks.
            * Testing on encrypted or compressed traffic data for broader applicability.