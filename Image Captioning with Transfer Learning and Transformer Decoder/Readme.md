# 

This project implements an advanced image captioning model using a hybrid architecture: an EfficientNetV2 backbone for image feature extraction, followed by a Transformer Encoder-Decoder for sequence generation. The model is trained and evaluated on the Flickr8k dataset, with capabilities for scaling to Flickr30k.

## **Table of Contents**

* [Project Description](#bookmark=id.n2bl2qve4hpi)  
* [Features](#bookmark=id.mpza4om3xcz6)  
* [Setup and Installation](#bookmark=id.5uakhc1glrxi)  
* [Dataset](#bookmark=id.bifo19cedi9n)  
* [Model Architecture](#bookmark=id.yirn0qgxoogy)  
* [Training](#bookmark=id.jhmk9iwr82o7)  
* [Evaluation Metrics](#bookmark=id.zg7sma8lpr16)  
* [Results](#bookmark=id.agiuy11kksd7)  
* [Scope for Future Improvements](#bookmark=id.1ll8zs6q5jjf)  
* [Usage](#bookmark=id.a7bvlpupcmph)  
* [Acknowledgements](#bookmark=id.q3va3esc35gq)

## **Project Description**

This project focuses on building an end-to-end deep learning model capable of generating descriptive captions for images. It leverages a pre-trained EfficientNetV2S as the convolutional neural network (CNN) backbone for robust image feature extraction. These features are then fed into a custom Transformer-based Encoder, which further processes the visual information. Finally, a Transformer-based Decoder generates the captions word by word, conditioned on the encoded image features and previously generated words.

## **Features**

* **Hybrid Architecture:** Combines EfficientNetV2S (CNN) with a Transformer Encoder-Decoder.  
* **Transfer Learning:** Utilizes a pre-trained EfficientNetV2S for efficient feature extraction.  
* **Fine-tuning:** Allows fine-tuning of the latter layers of the EfficientNetV2S backbone for task-specific adaptation.  
* **Custom Learning Rate Schedule:** Implements a linear warmup schedule for stable training.  
* **AdamW Optimizer:** Uses AdamW for optimized training with decoupled weight decay.  
* **Gradient Clipping:** Incorporates clipnorm to prevent exploding gradients in Transformers.  
* **Early Stopping:** Prevents overfitting by monitoring validation loss.  
* **Robust Data Pipeline:** Uses tf.data for efficient data loading, augmentation, and preprocessing.  
* **COCO Evaluation:** Integrates pycocoevalcap for comprehensive metric evaluation (BLEU, METEOR, ROUGE, CIDEr, SPICE).

## **Setup and Installation**

To set up the project locally or in a Colab environment, follow these steps:

1. **Clone the coco-caption repository:**  
   git clone https://github.com/tylin/coco-caption.git  
2. **Install pycocoevalcap and pycocotools from the cloned repository:**  
   cd coco-caption && pip install .  
   \# Then navigate back to your project root if necessary  
   \# cd ..  
3. Install required Python packages:  
   It's recommended to use a virtual environment.  
   pip install \-r requirements.txt  
4. **Download NLTK data:**  
   import nltk  
   nltk.download('punkt')  
   nltk.download('wordnet')

## **Dataset**

The model is primarily demonstrated using the **Flickr8k Dataset**.

* **Flickr8k:** Contains 8,000 images, each with 5 human-annotated captions.  
  * Images: Flickr8k\_Dataset.zip  
  * Captions: Flickr8k\_text.zip (containing Flickr8k.token.txt)

To use the dataset, ensure you have downloaded and unzipped the necessary files into your project directory. The load\_captions\_data function expects Flickr8k.token.txt and the Flicker8k\_Dataset folder.

For Flickr30k:  
The code structure supports an upgrade to the Flickr30k Dataset for potentially better results due to its larger size (30,000 images). This would require downloading Flickr30k images and its corresponding results\_20130124.token file and adapting the load\_captions\_data function accordingly.

## **Model Architecture**

The ImageCaptioningModel comprises:

1. **CNN Backbone (get\_cnn\_model):**  
   * EfficientNetV2S pre-trained on ImageNet.  
   * Initially frozen, with the last 20 layers (excluding Batch Normalization) unfrozen for fine-tuning.  
   * Output is global average pooled and reshaped to (batch\_size, 1, features).  
2. **Transformer Encoder (TransformerEncoderBlock):**  
   * Takes the image features as input.  
   * Consists of Multi-Head Attention, Layer Normalization, and Feed-Forward Networks.  
3. **Transformer Decoder (TransformerDecoderBlock):**  
   * Takes the encoded image features and partial captions as input.  
   * Utilizes self-attention (with causal masking) and cross-attention to the encoder outputs.  
   * Includes Positional Embeddings for sequence order.  
   * Outputs a probability distribution over the vocabulary for the next token.

## **Training**

The training process is configured with:

* **Optimizer:** keras.optimizers.AdamW with gradient clipping (clipnorm=2).  
* **Learning Rate Schedule:** Custom LRSchedule with linear warmup and a base learning rate (e.g., 5e-5).  
* **Loss Function:** keras.losses.SparseCategoricalCrossentropy(from\_logits=False, reduction="none").  
* **Early Stopping:** Monitors val\_loss with a patience of 5 epochs and restores the best weights.  
* **Data Augmentation:** keras.Sequential with RandomFlip, RandomRotation, and RandomContrast applied to images during training.

## **Evaluation Metrics**

The model's performance is evaluated using standard image captioning metrics via the pycocoevalcap library:

* **BLEU (1-4):** Bilingual Evaluation Understudy. Measures n-gram overlap between generated and reference captions.  
* **METEOR:** Metric for Evaluation of Translation with Explicit Ordering. Considers exact, stem, synonym, and paraphrase matches.  
* **ROUGE\_L:** Recall-Oriented Understudy for Gisting Evaluation (L-sum). Measures longest common subsequence.  
* **CIDEr:** Consensus-based Image Description Evaluation. Measures consensus with human judgments.  
* **SPICE:** Semantic Propositional Image Caption Evaluation. Measures semantic content.

## **Results**

After training and fine-tuning, the model achieved the following scores on a subset of the validation dataset (approximately 200 images, or as configured):

* **Bleu\_1:** 0.5733  
* **Bleu\_2:** 0.3091  
* **Bleu\_3:** 0.2597  
* **Bleu\_4:** 0.1685  
* **METEOR:** 0.2021  
* **ROUGE\_L:** 0.4337  
* **CIDEr:** 0.5092  
* **SPICE:** 0.1431

## **Scope for Future Improvements**

To further enhance the model's performance and capabilities, consider the following improvements:

* **Larger Datasets:** Train the model on more extensive datasets like COCO2017 or Flickr30k. Larger datasets provide more diverse image-caption pairs, leading to better generalization and caption quality.  
* **Vision Transformer (ViT) for Preprocessing:** Explore replacing the EfficientNetV2S backbone with a pure Vision Transformer (ViT) model for image feature extraction. ViTs have shown strong performance in vision tasks and could potentially capture richer visual representations for captioning.

## **Usage**

The project is structured as a Python script/Colab notebook.

1. **Run all cells:** Execute the notebook cells sequentially to:  
   * Install dependencies.  
   * Download and preprocess the dataset.  
   * Build and compile the model.  
   * Train the model.  
   * Run a full evaluation on the validation set.  
   * Generate sample predictions for random images.  
2. **Modify parameters:** Adjust hyperparameters like EPOCHS, BATCH\_SIZE, IMAGE\_SIZE, VOCAB\_SIZE, EMBED\_DIM, FF\_DIM, learning rate, and early stopping patience as needed in the "Setup" and "Model training" sections.  
3. **Generate Captions:** Use the generate\_caption() function to see qualitative results for random images.

## **Acknowledgements**

This project builds upon concepts and code derived from various Keras and TensorFlow tutorials on image captioning and Transformer architectures.