# D.A.N.N. (Dermatologic Analysis with Neural Networks)

![D.A.N.N. Banner](https://link_to_image/banner.png)

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
The following datasets were used for training, validation, and testing:

1. **[Base Dataset](https://www.kaggle.com/datasets/devdope/skin-disease-lightweight-dataset):**
   - Compiled from multiple sources, including HAM10000, Dermnet NZ, and Google Images.
   - 1,200+ training images.

2. **[Segmented Dataset](https://www.kaggle.com/datasets/devdope/skin-disease-lightweight-dataset-segmented):**
   - Processed using HSV and SlimSAM segmentation techniques.

3. **[Synthetic Augmentation Dataset](https://www.kaggle.com/datasets/devdope/synthetic-skin-disease-dataset):**
   - Generated with GANs using Stable Diffusion and custom LoRAs for Chickenpox, Measles, Herpes, and Monkeypox.

4. **[Combined Dataset](https://www.kaggle.com/datasets/devdope/skin-disease-lightweight-dataset-combined):**
   - Integrates real and synthetic images for improved model generalization.

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
| Model            | Dataset                        | Validation Accuracy | Sensitivity | Specificity |
|-------------------|--------------------------------|----------------------|-------------|-------------|
| ViT Transformer   | Base + Synthetic Augmentation | 92.77%              | 0.94        | 0.97        |
| DinoV2            | Base + Synthetic Augmentation | 89.95%              | 0.89        | 0.96        |
| Swin Transformer  | Base + Synthetic Augmentation | 90.36%              | 0.91        | 0.95        |

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
- Erick Garcia (Lead Developer & Researcher)  
- [Additional Team Members]

---

## **Acknowledgments**
Special thanks to **CivitAI** for hosting the LoRAs used in this project and **Hugging Face Spaces** for providing an accessible deployment platform.

---

## **References**
For a complete list of references and citations, refer to the [article](https://link_to_published_article).

---

## **Contact**
For questions or collaborations, feel free to reach out:  
📧 dev.garcia.espinosa.erick@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/dev-ai-erick)  
