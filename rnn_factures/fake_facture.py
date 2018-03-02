from faker import Faker
from faker.providers import BaseProvider
import random
from babel.dates import format_date
from tqdm import tqdm
import numpy as np
from babel.dates import format_date
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt

# change this if you want it to work with another language
LOCALES = ['en_US']

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           'dd/MM/YYYY',
           'dd-MM-YYYY',
           'dd-MM-YY',
            'MM-dd-YYYY',
           'MM/dd/YYYY',
           'YYYY MMM dd',
           'ddMMYYYY']

fake = Faker('en_US')
# name= fake.name()
# address = fake.address()
# date = fake.date()
# my_word_list = [name, address, date]
# print(fake.sentence(ext_word_list=my_word_list))

def load_date():
    """
        Loads some fake dates
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),
                                      locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

    except AttributeError as e:
        return None, None, None
    return human_readable, machine_readable, dt

class MyProvider(BaseProvider):
    def facture(self):
        name= fake.name()
        address = fake.address()
        input_date,output_date,_ = load_date()
        amount = str(round(random.uniform(1,500),2))
        #currency = ''
        return name+" "+input_date+" TOTAL "+amount+" € "+address, output_date

"""
fake.add_provider(MyProvider)
factures = []

for i in range(10):
    facture, output_date = fake.facture()
    factures.append(facture)

print(factures)
"""
def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
        human is the dictionary of input characters
        machine is the dictionary of output characters
        dataset is a list of tuple(facture, facture_date)
    """
    Tx = 0
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    fake.add_provider(MyProvider)

    for i in tqdm(range(m)):
        facture,facture_date = fake.facture()
        if facture is not None:
            if Tx < len(set(facture)):
                Tx = len(set(facture))
            dataset.append((facture, facture_date))
            human_vocab.update(tuple(facture))
            machine_vocab.update(tuple(facture_date))

    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'],
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v: k for k, v in inv_machine.items()}

    return dataset, human, machine, inv_machine,Tx

def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"

    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"

    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """

    # make lower to standardize
    string = string.lower()
    string = string.replace(',', '')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    # for x in string:
    #     print('x: {} and vocab: {}'.format(x,vocab.get(x, '<unk>')))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    # print (rep)
    return rep


def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary

    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters

    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """

    l = [inv_vocab[i] for i in ints]
    return l

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]
    #print('X ',X)
    #print('Y ',Y)
    # for x in X:
    #     print('x',x)
    #     print(to_categorical(x,num_classes=len(human_vocab)))
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))
    # print('Xoh ',Xoh)
    # print('Yoh',Yoh)

    return X, np.array(Y), Xoh, Yoh
