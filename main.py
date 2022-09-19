# libraries
import tensorflow as tf 
import re 
import argparse

#inputs 
file='./dan.txt'

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

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def translator(sentence, encoder, decoder,inp_lang,max_length_inp,
              targ_lang,max_length_targ):
    sentence = text_processing(sentence, False)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp[1],
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    enc_out, enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ[1]):
        predictions, dec_hidden = decoder(dec_input,dec_hidden,enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if targ_lang.index_word[predicted_id] != '<end>':

            result += targ_lang.index_word[predicted_id] + ' '
            dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def go(args):
    
    fore_tensor, english_tensor,fore_tokenizer,english_tokenizer = create_data(file)
    max_length_fore, max_length_english = fore_tensor.shape, english_tensor.shape
    encoder = tf.keras.models.load_model('saved_model/encoder')
    decoder = tf.keras.models.load_model('saved_model/decoder')
    result, sentence = translator(args.input_sentence, encoder,decoder,
                                 fore_tokenizer,max_length_fore,english_tokenizer,
                                 max_length_english)
    print('given sentence: %s' % (sentence))
    print('translation: {}'.format(result))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_sentence",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)