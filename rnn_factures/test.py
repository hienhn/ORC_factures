from fake_facture import *

dataset, human_vocab, machine_vocab, inv_machine_vocab,Tx = load_dataset(10000)
print("First 10 factures: ", dataset[:10])
print('input dictionary: ',human_vocab)
print('output dictionary: ',machine_vocab)
#Tx = 45
print('Tx', Tx)
Ty = 10
#string = "Brioche Doree 01 MARCH 2017 TOTAL 10€ 34 avenue Opera 75001 Paris"
#string_to_int(string,Tx,human_vocab)
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
#preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print("first 10 factures in vecteur representation: ",X[:10,])
print("first 10 factures in matrice representation: ",Xoh[:10,])