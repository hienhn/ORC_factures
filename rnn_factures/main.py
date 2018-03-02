from RNN_model import *

def main():
    model1 = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    print(model1.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    #model1.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
    model1.fit([Xoh, s0, c0], outputs, epochs=50, batch_size=1000)

    # serialize weights to HDF5: need to install h5py
    model1.save_weights("model1.h5")
    # print("Saved model to disk")
    #model1.load_weights('model.h5')

    # evaluate the model
    scores = model1.evaluate([Xoh,s0,c0], outputs)
    print("Score \n%s: %.2f%%" % (model1.metrics_names[1], scores[1] * 100))

    EXAMPLES = ['Brioche Doree 01/03/2017 TOTAL 10€ 34 avenue Opera 75001 Paris']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
        source = np.expand_dims(source, axis=0)
        # print(source.shape)
        prediction = model1.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output))
if __name__ == "__main__":
    main()
