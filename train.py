# libraries
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
import re

# defining loss function and optimizer 
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                            reduction='none')

#inputs 
file='./dan.txt'
batch_size=40
epochs=10
embedding_dim = 256
units = 1024

#functions
def text_processing(w,english):
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    if english is True:
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=w.lower()
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def read_data(file):
    en=[]
    no=[]
    with open(file,'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines: 
            sen=line.split("CC")[0]
            index = sen.index('\t')
            en_sen=sen[:index]
            en_sen=text_processing(en_sen,True)
            no_sen=sen[index+1:]
            no_sen=text_processing(no_sen,False)
            en.append(en_sen)
            no.append(no_sen)
    return no,en

def tokenizer(sen):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(sen)

    tensor = lang_tokenizer.texts_to_sequences(sen)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
    return tensor, lang_tokenizer

def create_data(file):
    inp_lang,out_lang = read_data(file)

    input_tensor, inp_lang_tokenizer = tokenizer(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenizer(out_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def loss_funct(real,pred):
    loss= loss_object(real,pred)
    return tf.reduce_mean(loss)

def train_step(inp, targ, encoder, decoder,targ_lang):
    loss = 0

    with tf.GradientTape() as tape: 
        
        enc_output, enc_hidden = encoder(inp)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden= decoder(dec_input, dec_hidden,enc_output )

            loss += loss_funct(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def train(dataset,epochs,vocab_eng_size, vocab_fore_size,embedding_dim, units,
          steps_per_epoch,targ_lang):
    encoder = Encoder(vocab_fore_size, embedding_dim, units)
    decoder = Decoder(vocab_eng_size, embedding_dim, units)
    loss = []

    for epoch in range(epochs):
        total_loss = 0

        for (batch,(inp,out)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, out, encoder, decoder,targ_lang)
            total_loss+=batch_loss
    
    loss.append(total_loss/steps_per_epoch)
    return encoder, decoder, loss


if __name__ == "__main__":
    
    # obtaining tokenizer and embeddings
    fore_tensor, english_tensor,fore_tokenizer,english_tokenizer = create_data(file)
    max_length_fore, max_length_english = fore_tensor.shape, english_tensor.shape
    
    # creating the dataset
    dataset = tf.data.Dataset.from_tensor_slices((fore_tensor,english_tensor)).shuffle(len(english_tensor))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    vocab_eng_size = len(english_tokenizer.word_index)+1
    vocab_fore_size = len(fore_tokenizer.word_index)+1
    steps_per_epoch=len(english_tensor)//batch_size
    # training function
    encoder, decoder, loss = train(dataset,epochs,vocab_eng_size, 
                                 vocab_fore_size,embedding_dim, units, 
                                 steps_per_epoch,english_tokenizer)
    encoder.save('saved_model/encoder')
    decoder.save('saved_model/decoder')
