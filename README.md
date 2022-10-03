# NLP-seq2seq-Danish-to-English
This repository includes a neural machine translation system based on the sequence-to-sequence attention model (the code for implementing the seq-to-seq attention model in this repository is inspired from [1] ) for translating Danish phrases to English. 
The dataset is available from the ManyThings.org website [2] with examples drawn from the Tatoeba Project. 
The dataset is comprised of Danish phrases and their English counterparts. 
# How to run 
To train the model, please run: 
**python train.py** 

The commond trains the model and saves the encoder and decoder. Then, to translate the danish phrases run:
**python main.py "your_phrase"**

The commond generates the translation of the phrase to English. 
Script train.py contains codes to translate Danish phrases, and train.py includes codes in Tnesorflow to train the model. 
# Result
The model is able to translate correctly phrases that are comparatively close to phrases existed in training data. 
For instance, the model translates jeg er glad i dag to I am happy today, which is true. The other examples: 

hun elskede ham > she loved him.

Jeg arbejder i dag > I am working today.

hun er rig > she is rich.

Jeg ventede på en bus > I was waiting for a bus. 

er du skør > you are crazy. 

To improve the model, multiple GRUs and a larger dataset can be used.  
# References
[1]. https://www.tensorflow.org/text/tutorials/nmt_with_attention#test_the_attention_layer

[2]. http://www.manythings.org/anki/
