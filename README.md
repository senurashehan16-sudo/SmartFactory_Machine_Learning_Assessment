# SmartFactory Predictive Maintenance and Visual Inspection

**Student:** Senura Shehan  
**Module:** 6FTC2175 Advanced Machine Learning and Neural Network  
**Assessment:** Assignment 1

## Project overview

This project compares three machine learning approaches for a SmartFactory:

- **Random Forest:** predicts machine failure from sensor measurements.
- **Feed-forward neural network:** predicts machine failure using the same sensor inputs and test rows.
- **Convolutional neural network (CNN):** checks casting photographs for visible defects.

The sensor models and CNN have different targets. Their accuracy scores should not be treated as results on the same task.

## Main files

| File | Purpose |
| --- | --- |
| `Senura_Shehan_Advance_Machine_Learning_Assessment_01.ipynb` | Main notebook containing the code, saved outputs and commentary. |
| `SmartFactory_Model_Comparison_Report.pdf` | Report comparing the models and explaining deployment choices. |
| `ai4i2020.csv` | Sensor dataset, if included in the repository. It can also be downloaded using the link below. |
| `README.md` | Project summary, dataset links and instructions for running the notebook. |

The notebook copy with added explanations can also be used. If that copy is uploaded, open its `.ipynb` file in Colab instead.

## Dataset links

### AI4I 2020 Predictive Maintenance Dataset

- **Source:** UCI Machine Learning Repository.
- **Dataset page:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Dataset DOI:** https://doi.org/10.24432/C5HS5C
- **Required file:** `ai4i2020.csv`.

This synthetic dataset contains 10,000 records. The notebook uses five numerical process measurements and product type to predict `Machine failure`. Identifiers and the five failure-type flags are removed from the inputs. The dataset is split into 70% training and 30% testing, keeping similar class proportions.

### Casting Product Image Data for Quality Inspection

- **Source:** Kaggle.
- **Dataset page:** https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- **Required image folder:** `casting_512x512`.
- **Class subfolders:** `ok_front` and `def_front`.

Download the dataset ZIP from the Kaggle page. A Kaggle account may be needed. The notebook searches the extracted ZIP for the original `casting_512x512` folder. Use the ZIP containing that folder rather than a ZIP containing only the other image versions.

The experiment uses 1,300 original images: 519 OK and 781 defective. The split contains 910 training, 195 validation and 195 test images. The images are resized to 128 by 128 in grayscale.

The large image ZIP does not need to be stored in this repository; download it from the source above before running the image experiment.

## How to run in Google Colab

1. Download the final `.ipynb` notebook from this repository.
2. Open [Google Colab](https://colab.research.google.com/) and upload the notebook.
3. Choose a GPU runtime for the CNN experiment if one is available. The saved notebook metadata records a T4 GPU configuration.
4. Run the notebook from the top, in order.
5. At the first upload prompt, select `ai4i2020.csv`.
6. Continue through the sensor preprocessing, Random Forest, feed-forward network and their evaluation plots.
7. At the image-dataset upload prompt, select exactly one dataset ZIP containing `casting_512x512`.
8. Continue through the image split, loading pipeline, CNN training, learning curves and final test evaluation.

The main image workflow uses the uploaded ZIP. The Google Drive examples that appear as text cells are optional references and are not required for this workflow.

The notebook uses these Python libraries:

- NumPy
- pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- seaborn

These libraries are generally available in Colab. For a local Jupyter environment, the interactive Colab upload cells would need to be replaced with local file paths.

## Notebook structure

### Part A Classical and feed-forward models

- Check the sensor data and missing values.
- Remove identifiers and target-related failure flags.
- Split the data and scale numerical measurements using the training partition.
- Encode product type and align training and test columns.
- Train a Random Forest with 100 trees and balanced class weights.
- Train a small feed-forward network with hidden layers of 16 and eight units, dropout and early stopping.
- Evaluate both models on the same 3,000 test rows.
- Review confusion matrices, feature importance, ROC curves and precision-recall curves.

### Part B Convolutional neural networks

- Load, label and split the casting images.
- Train both CNNs from scratch, without pretrained weights.
- Compare V1 with filters `[16, 32]` against V2 with filters `[16, 32, 64]`.
- Compare validation loss, accuracy, defect F1 and parameter counts.
- Select the highest validation defect F1, using lower validation loss to break a tie.
- Evaluate the selected CNN on the held-out image test set.

### Part C Critical comparison

The accompanying PDF report compares interpretability, data needs, computational requirements, training cost and predictive performance. It recommends a monitored pilot using Random Forest for maintenance warnings and CNN V2 for visual inspection, with further validation before wider deployment.

## Saved test results

Precision, recall and F1 below refer to the positive class: **machine failure** for the sensor models and **defective product** for the CNN.

| Model | Task | Test samples | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | Machine failure | 3,000 | 97.73% | 86.96% | 39.22% | 54.05% |
| Feed-forward network | Machine failure | 3,000 | 89.17% | 21.63% | 83.33% | 34.34% |
| CNN V2 | Casting defect | 195 | 96.41% | 98.25% | 95.73% | 96.97% |

Random Forest missed 62 failures and raised six false alarms. The feed-forward network missed 17 failures but raised 308 false alarms. The CNN's reported scores and class counts imply five missed defects and two false rejects.

CNN V2 achieved 94.87% validation accuracy and 95.73% validation defect F1, compared with 89.74% and 91.38% for V1. V2 used 547,649 parameters, compared with 1,053,441 for V1.

These are saved results from one run per model. They are not guarantees for future training runs or real factory use. The image workflow uses seed 42, but the feed-forward training does not explicitly set a random seed before creating that model. Software versions and hardware can also affect results.

## Limitations

- Machine failure and finished-product defects are different outcomes.
- The sensor and image records are not paired, so combined system accuracy was not measured.
- The sensor test has only 3.4% failures; accuracy alone can be misleading.
- The image test contains only 195 photographs.
- Random Forest fitted 7,000 rows; the network reserved 1,400 of those rows for validation and fitted 5,600.
- Factory prediction latency and financial costs were not measured.
- Real production data and error-cost information are needed before deployment.

## Data acknowledgement

The datasets belong to their original providers. The UCI AI4I dataset is available under CC BY 4.0. See each dataset's source page for attribution and usage terms.
