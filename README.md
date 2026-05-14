# Binary Change Detection on EO-SAR Image Pairs
### GalaxEye Space - AI Research Intern - Technical Assignment

A U-Net based binary change detection pipeline for co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) image pairs across multiple disaster events.

## Project Description:
This project implements an end-to-end binary change detection pipeline that classifies each pixel in a satellite image pair as either 
**Changed (1)** or **Unchanged (0)**. The approach adopts early fusion of EO and SAR modalities via channel concatenation into a 4-channel 
input, processed by a U-Net encoder-decoder architecture trained with a combined Dice and weighted BCE loss to address severe class imbalance.

## Requirements:

- Python 3.10+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- matplotlib
- seaborn
- scikit-learn
- tqdm
- Google Colab (recommended for GPU access)

Install all dependencies:
pip install -r requirements.txt

## Environment Setup:

This project is designed to run on **Google Colab** with GPU enabled.

1. Open `Atri_Kar_GalaxEye.ipynb` in Google Colab
2. Go to **Runtime → Change Runtime Type → GPU (T4)**
3. Run Cell 0 to configure paths
4. Run Cell 2 to mount Google Drive
5. Place dataset zip files in your Google Drive folder
6. Update `DRIVE_FOLDER` in Cell 0 to match your folder path

## Dataset Structure:

After running Cell 3 (unzip), the dataset will be structured as:

dataset/

    train/

        pre-event/    ← EO optical images (RGB, 1024x1024)
        
        post-event/   ← SAR grayscale images (1024x1024)
        
        target/       ← Binary masks (0=No-Change, 1=Change)
    
    val/
    
        pre-event/
        
        post-event/
        
        target/
    
    test/
    
        pre-event/
       
        post-event/
        
        target/

**Note:** Use only the provided dataset. External remote sensing data is not permitted for training or fine-tuning.

## Training:

To train from scratch:

Run cells in order: Cell 0 → Cell 9, then Cell 10

All hyperparameters are logged in config.yaml

To resume training from a saved checkpoint:
```python
# In Cell 10, set:
RESUME = True
START_EPOCH = 20      # Last completed epoch
BEST_VAL_LOSS = 0.5179  # Best val loss from last run
```

## Evaluation:

To evaluate using saved model weights:

Run cells in order: Cell 0 → Cell 9, then Cell 11 (Load Weights)
Then run Cell 12 for evaluation metrics
Then run Cell 13 for qualitative visualisations

Or with a custom checkpoint:
```python
# In Cell 11, update:
best_model_path = "/path/to/your/checkpoint.pth"
```

## Model Weights:

The final trained model checkpoint is available for download:

🔗 **[Download best_model.pth](https://huggingface.co/atrikar/galaxeye-change-detection/resolve/main/best_model.pth)**

Place the downloaded file at the path specified in `MODEL_PATH` 
in Cell 0, then run Cell 11 to load and evaluate.


## Results:

All metrics computed for the **Change class (label = 1)**:

| Metric | Validation | Test |
|--------|-----------|------|
| IoU | 0.2732 | 0.0052 |
| Precision | 0.3219 | 0.0053 |
| Recall | 0.6437 | 0.1282 |
| F1 Score | 0.4292 | 0.0103 |

The significant performance gap between validation and test sets 
is attributed to domain shift. The test split contains 
geographically distinct scenes (scene_09) absent from training data.


## Citation / References:

- Ashok, H. G. (2014). Survey on Change Detection in SAR Images. International Journal of Computer Applications.
  
- Bandara, W. G. C., & Patel, V. M. (2022). A Transformer-Based Siamese Network for Change Detection (ArXiv:2201.01293). arXiv. https://doi.org/10.48550/arXiv.2201.01293
  
- Daudt, R. C., Saux, B. L., & Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection (ArXiv:1810.08462). arXiv. https://doi.org/10.48550/arXiv.1810.08462

- Frost, V. S., Stiles, J. A., Shanmugan, K. S., & Holtzman, J. C. (1982). A Model for Radar Images and Its Application to Adaptive Digital Filtering of Multiplicative Noise. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-4(2), 157–166. https://doi.org/10.1109/TPAMI.1982.4767223

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT press.

- Jiang, X., Li, G., Liu, Y., Zhang, X.-P., & He, Y. (2020). Change Detection in Heterogeneous Optical and SAR Remote Sensing Images Via Deep Homogeneous Feature Fusion. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 1551–1566. https://doi.org/10.1109/JSTARS.2020.2983993

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization (ArXiv:1412.6980). arXiv. https://doi.org/10.48550/arXiv.1412.6980

- Lee, J.-S. (1980). Digital Image Enhancement and Noise Filtering by Use of Local Statistics. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-2(2), 165–168. https://doi.org/10.1109/TPAMI.1980.4766994

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. 2017 IEEE International Conference on Computer Vision (ICCV), 2999–3007. https://doi.org/10.1109/ICCV.2017.324

- Mañas, O., Lacoste, A., Giro-i-Nieto, X., Vazquez, D., & Rodriguez, P. (2021). Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2103.16607

- Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation (ArXiv:1606.04797). arXiv. https://doi.org/10.48550/arXiv.1606.04797

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation (ArXiv:1505.04597). arXiv. https://doi.org/10.48550/arXiv.1505.04597

- Saidi, S., Idbraim, S., Karmoude, Y., Masse, A., & Arbelo, M. (2024). Deep-Learning for Change Detection Using Multi-Modal Fusion of Remote Sensing Images: A Review. Remote Sensing, 16(20), 3852. https://doi.org/10.3390/rs16203852

- Tuia, D., Persello, C., & Bruzzone, L. (2016). Domain Adaptation for the Classification of Remote Sensing Data: An Overview of Recent Advances. IEEE Geoscience and Remote Sensing Magazine, 4(2), 41–57. https://doi.org/10.1109/MGRS.2016.2548504



## Author:

**Atri Kar**

atrikar77@gmail.com

[LinkedIn](https://linkedin.com/in/atrikar/)

[GitHub](https://github.com/atri-kar)
