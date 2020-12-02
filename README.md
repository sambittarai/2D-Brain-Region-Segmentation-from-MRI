# UNET_Segmentation

Segmentation of MRI images using UNET

## Requirements:
  * Python >= 3.5
  * Keras

## Segmentation of MRI Images

**MRI Image and it's corresponding Segmented Image**      
![](Training_Images/MRI_Img.png)      ![](Training_Images/Segmented_Img.png)

**One hot encoding of the Masks**

![](Segmented_Masks/Mask1.png)  ![](Segmented_Masks/Mask2.png)  ![](Segmented_Masks/Mask3.png)  ![](Segmented_Masks/Mask4.png)



**UNET Architecture**
Using the UNET model to do the segmentation.
UNET is an Autoencoder with skip connections.
![](UNET_Architecture.png)



**Segmented Image Prediction**


**Prediction**

![](Predicted_Images/Prediction_96%25.png)

**Actual Segmented Image**

![](Predicted_Images/Original.png)


**Loss & Accuracy curves**

![](Loss%26Acc_Curves/acc.png)           ![](Loss%26Acc_Curves/loss.png)


