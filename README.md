# English-to-French-transformers
English to French machine translation models using: `Transformers implementation`, `LLM fine-tuning` and `LSTMs`.
## Transformer from scratch
The transformer architecture was implemented completely in the `mt-transformer from scratch.ipynb` file and yielded the following results:
- loss of 0.5567
- accuracy of 90%
## Fine-tuning a pre-trained transformer model
Fine-tuning the **Helsinki Model** on HuggingFace and can be accessed through this <a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-fr"> link</a>  
*Dataset*: <a href="https://huggingface.co/datasets/Helsinki-NLP/kde4">KDE4 Dataset</a>  
The model was fine-tuned using the following parameters:
- 3 epochs
- 5e-5 learning rate
- 0.01 weight decay rate

It yielded the following results:
* BLEU score: 4.799920346318993 
* Precisions:7.220956167571761,5.3723560517038775,4.16707871835443, 3.2835597283260087
The fine-tuned version of the model can be found here <a href="https://huggingface.co/ranwakhaled/helsinki-finetuned-en-to-fr">https://huggingface.co/ranwakhaled/helsinki-finetuned-en-to-fr</a>
## LSTM model
### Model Architecture & Evaluation 
Traditional Model: LSTM using TensorFlow 
* The model was trained using the whole dataset 
* Training data was split into train and testing with a ratio of **80%:20%**   
The model had **2** LSTM layers each **128** units.
### Metrics
*Training results*  
- Best training loss:  0.6122 
- Best validation loss: 0.8343

*Testing results*  
  
BLEU score was used to evaluate the model getting a score of 0.04715, which is not that good however, Even though the BLEU score is low, the predictions make a lot of sense and since it measures overlapping words sometimes we can reach the same meaning with different words so the human evaluation in our case is better
