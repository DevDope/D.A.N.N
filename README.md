# D.A.N.N. (Dermatologic Analysis with Neural Networks)

![D.A.N.N. Banner](https://thumbs2.imgbox.com/25/57/vcnPG4No_t.pn)

## **Overview**
D.A.N.N. is a lightweight, portable device designed to assist healthcare professionals in resource-limited environments. Using a **Vision Transformer (ViT)** model enhanced with synthetic data generated via **Stable Diffusion GAN**, D.A.N.N. achieves state-of-the-art performance in skin lesion classification, with a validation accuracy of **92.77%**.

The device is built on **Raspberry Pi 4** hardware and provides offline functionality, making it ideal for **first-level clinics** in rural areas.

---

## **Key Features**
- **Portable Device:** Powered by a Raspberry Pi 4 with an integrated 3.5-inch touchscreen.
- **Transformer-Based Models:** Includes ViT, DinoV2, and Swin Transformers for skin lesion classification.
- **Synthetic Data Augmentation:** Enhances performance using GANs and custom LoRAs for underrepresented classes.
- **Explainable AI:** Implements LIME for interpretable predictions, critical for medical applications.
- **Hybrid Deployment:** Accessible both offline (device) and online (Hugging Face Spaces).

---


## **Datasets**

This section provides a detailed overview of the datasets used and created for this research. These datasets include the base dataset, segmented datasets, and datasets augmented with synthetic images.

---

### **1. Base Dataset**
- A comprehensive collection of images of skin diseases compiled from multiple sources:
  - **HAM10000 (2019)**, Kaggle, Google Images, Dermnet NZ, Bing Images, Yandex, the Hellenic Dermatological Atlas, and the Dermatological Atlas.
- Curated to include diverse skin conditions, this dataset serves as the foundation for all subsequent datasets and preprocessing steps.
- **Publicly available for download:** [Base Dataset on Kaggle](https://www.kaggle.com/datasets/devdope/skin-disease-lightweight-dataset).

---

### **2. Segmented Dataset**
- Created by applying the following image segmentation methods to the base dataset:
  - **HSV Color Space Segmentation:** Images were converted to the HSV color space. Lesion areas were isolated using thresholds:
    - **Hue (H):** [0, 60]
    - **Saturation (S):** >90
    - **Value (V):** [10, 200]
  - **Adaptive Thresholding:** After HSV segmentation, images were converted to grayscale, and adaptive thresholding was applied with:
    - **Block size:** 11
    - **Constant value:** 2
- **Publicly available for download:** [Segmented Dataset on Kaggle](https://www.kaggle.com/datasets/devdope/skin-disease-lightweight-dataset-segmented).

---

### **3. Segmented Dataset Using SlimSAM**
- Generated using the **"Zigeng/SlimSAM-uniform-50" model**, applied to the base dataset.
- **Enhancements:**
  - Image contrast was increased before segmentation to improve the model's accuracy.
  - SlimSAM performed precise automatic segmentation of lesion areas.
- **Publicly available for download:** [SlimSAM Segmented Dataset on Kaggle](https://www.kaggle.com/datasets/devdope/skin-disease-variations-dataset).

---

### **4. Mixed Segmented Dataset**
- Combines images from both **HSV segmentation** and **SlimSAM segmentation** approaches.
- Random selection ensures a balanced representation of both methods, providing diversity for model training and evaluation.
- **Publicly available for download:** [Mixed Segmented Dataset on Kaggle](https://www.kaggle.com/datasets/devdope/skin-disease-variations-dataset).

---

### **5. Base + Synthetic Augmentation Dataset**
- Incorporates synthetic images generated to enhance the base dataset, specifically for underrepresented categories:
  - **Herpes, Measles, Chickenpox, and Monkeypox.** Melanoma was excluded due to the abundance of existing data.
- **Generation Details:**
  - **LoRAs Training:** 60 epochs and 30,000 steps using OneTrainer on the base dataset.
  - **Image Generation:** Generated using the Fooocus API, guided by a trained discriminator model to ensure correct categorization.
  - **Pre-trained Model:** **JuggernautXL V7** (Stable Diffusion XL) served as the foundation for LoRAs. Weights and configurations are available on [CivitAI](https://civitai.com/user/DevDope).
- **Publicly available for download:** [Synthetic Augmentation Dataset on Kaggle](https://www.kaggle.com/datasets/devdope/synthetic-skin-disease-datasetreal-and-synthetic).

---

### **6. Base + Synthetic Augmentation for Classes with Fewer Data**
- Focuses on balancing the dataset by generating synthetic images exclusively for underrepresented categories:
  - **Chickenpox** and **Measles.**
- Only original images for other categories were retained to ensure balance.
- **Publicly available for download:** [Synthetic Augmentation for Fewer Data on Kaggle](https://www.kaggle.com/datasets/devdope/synthetic-skin-disease-datasetonly-synthetic).

---


---

## **How to Use**
### Online (Hugging Face Space)
Test the model interactively here:  
👉 [D.A.N.N. on Hugging Face Spaces](https://huggingface.co/spaces/DopeErick/SkinLesionClassifierSpace)

### Offline (Device Setup)
1. **Hardware Requirements:**
   - Raspberry Pi 4 with 8GB RAM
   - Raspberry Pi Camera Module 3
   - 3.5-inch touchscreen
2. **Software Installation:**
   - Install dependencies: PyTorch, TensorFlow, Transformers, Gradio.
   - Load the trained model and configure the Gradio interface.
3. **Usage:**
   - Capture an image of the skin lesion.
   - View the prediction and confidence scores on the touchscreen interface.

---

## **Performance**
### **Performance Comparison by Model and Dataset**

| Model            | Dataset                                  | Train Accuracy | Validation Accuracy | Epochs |
|-------------------|------------------------------------------|----------------|----------------------|--------|
| **ViT Transformer** | Base                                    | 99.6%          | 90.76%              | 6      |
|                   | Segmented Dataset                       | 99.9%          | 87.81%              | 7      |
|                   | Segmented Dataset using SlimSAM         | 99.7%          | 67.87%              | 7      |
|                   | Mixed Segmented Dataset                 | 99.8%          | 88.35%              | 7      |
|                   | Base + Synthetic Augmentation Dataset   | 99.6%          | 92.77%              | 6      |
|                   | Base + Synthetic Augmentation (less classes) | 99.7%          | 89.15%              | 7      |
| **DinoV2**        | Base                                    | 96.4%          | 88.75%              | 35     |
|                   | Segmented Dataset                       | 96.3%          | 84.73%              | 35     |
|                   | Segmented Dataset using SlimSAM         | 94.0%          | 76.70%              | 35     |
|                   | Mixed Segmented Dataset                 | 96.5%          | 81.52%              | 35     |
|                   | Base + Synthetic Augmentation Dataset   | 96.8%          | 87.50%              | 35     |
|                   | Base + Synthetic Augmentation (less classes) | 96.1%          | 89.95%              | 35     |
| **Swin Transformer** | Base                                  | 96.7%          | 91.96%              | 20     |
|                   | Segmented Dataset                       | 96.0%          | 85.15%              | 20     |
|                   | Segmented Dataset using SlimSAM         | 93.5%          | 85.04%              | 20     |
|                   | Mixed Segmented Dataset                 | 95.9%          | 90.76%              | 20     |
|                   | Base + Synthetic Augmentation Dataset   | 98.2%          | 89.15%              | 20     |
|                   | Base + Synthetic Augmentation (less classes) | 96.7%          | 90.36%              | 20     |


---
## **Base Model(ViT) vs Syntetic Augmented Model(ViT)**
D.A.N.N. was evaluated using two models:
- **Model 1 (M1):** Trained on the base dataset without synthetic augmentation.
- **Model 2 (M2):** Trained on the base dataset augmented with synthetic data.

### **Disease-Level Metrics**
The table below summarizes precision, recall, F1-score, sensitivity, specificity, and AUC for each disease across both models:

| Disease        | Precision (M1) | Recall (M1) | F1 (M1) | Sensitivity (M1) | Specificity (M1) | AUC (M1) | Precision (M2) | Recall (M2) | F1 (M2) | Sensitivity (M2) | Specificity (M2) | AUC (M2) |
|----------------|----------------|-------------|---------|------------------|------------------|----------|----------------|-------------|---------|------------------|------------------|----------|
| Monkeypox      | 0.9318         | 0.8200      | 0.8723  | 0.8200           | 0.9849           | 0.9832   | 0.9565         | 0.8800      | 0.9166  | 0.8800           | 0.9899           | 0.9843   |
| Measles        | 0.8653         | 0.9000      | 0.8823  | 0.9000           | 0.9648           | 0.9822   | 0.9038         | 0.9400      | 0.9215  | 0.9400           | 0.9748           | 0.9887   |
| Chickenpox     | 0.8301         | 0.8979      | 0.8627  | 0.8979           | 0.9550           | 0.9713   | 0.8269         | 0.8775      | 0.8514  | 0.8775           | 0.9550           | 0.9696   |
| Herpes         | 0.9387         | 0.9200      | 0.9292  | 0.9200           | 0.9849           | 0.9922   | 0.9791         | 0.9400      | 0.9591  | 0.9400           | 0.9949           | 0.9943   |
| Melanoma       | 0.9803         | 1.0000      | 0.9900  | 1.0000           | 0.9949           | 1.0000   | 0.9803         | 1.0000      | 0.9900  | 1.0000           | 0.9949           | 1.0000   |

---

### **Observations**
1. **Monkeypox:**
   - Model 2 improved recall (from 0.82 to 0.88) and sensitivity (from 0.8200 to 0.8800), reducing the likelihood of missed cases while maintaining a high AUC of 0.9843.

2. **Measles:**
   - Recall and sensitivity showed notable improvements (from 0.90 to 0.94), with the AUC increasing from 0.9822 to 0.9887, reflecting better generalization and classification accuracy.

3. **Chickenpox:**
   - While precision slightly decreased (from 0.8301 to 0.8269), recall and sensitivity remained relatively stable, indicating moderate support for this class.

4. **Herpes:**
   - Significant gains in precision (from 0.9387 to 0.9791), recall (from 0.9200 to 0.9400), and sensitivity (from 0.9200 to 0.9400) highlight the robustness of Model 2 for this disease.

5. **Melanoma:**
   - Both models achieved exceptional metrics across all evaluation parameters, demonstrating robust classification performance.

---

### **ROC Curves and Confusion Matrices**
The performance of both models was further analyzed using ROC curves and confusion matrices:
- **ROC Curves:** Demonstrated improved trade-offs between sensitivity and specificity in Model 2, particularly for underrepresented classes like Monkeypox and Chickenpox.
- **Confusion Matrices:** Highlighted reduced misclassification rates in Model 2, especially for diseases with higher clinical significance.

![ROC and Confusion Matrices](https://link_to_image/roc_confusion.png)

---

### **Final Observations**
- Model 2 exhibited better sensitivity, precision, and recall across most diseases, demonstrating the benefits of synthetic data augmentation in improving classification consistency and accuracy.
- Inference time for Model 2 was reduced to **188.55 seconds**, compared to **194.38 seconds** for Model 1, showcasing improved computational efficiency.

---


## **Explainability**
Using LIME, the model highlights the most relevant regions for each prediction, ensuring transparency in medical diagnosis. Below is an example visualization:  
![LIME Visualization](https://link_to_image/lime.png)

---

## **Future Work**
- Test the device in real-world clinical settings.
- Explore larger transformer architectures for improved performance.
- Extend the dataset to include additional skin conditions.

---

## **Contributors**
- M.Sc. Erick Garcia
- Ph.D. Jose Sergio Ruiz Castilla
- Ph.D. Farid Garcia Lamont


---

## **Acknowledgments**
Special thanks to **CivitAI** for hosting the LoRAs used in this project and **Hugging Face Spaces** for providing an accessible deployment platform.

---

## **References**
For a complete list of references and citations, refer to the [article](https://link_to_published_article).(Coming Soon)

---

## **Contact**
For questions or collaborations, feel free to reach out:  
📧 dev.garcia.espinosa.erick@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/dev-ai-erick)  
