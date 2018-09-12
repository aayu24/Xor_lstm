import argparse
import numpy as np

from utils import build_model, generate_samples, model_plot

np.random.seed(42)

def main(length=40, num_epochs=20):
    '''
    Build and train LSTM network to solve XOR problem
    '''
    X_train, y_train, X_test, y_test = generate_samples(length=length)
    model = build_model()
    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        batch_size=32,
        validation_split=0.10,
        shuffle=False)

    # Evaluate model on test set
    preds = model.predict(X_test)
    preds = np.round(preds[:, 0]).astype('float32')
    acc = (np.sum(preds == y_test) / len(y_test)) * 100
    print('Accuracy: {:.2f}%'.format(acc))

    # Plotting loss and accuracy
    model_plot(history)
    return

if __name__ == '__main__':
    '''
    Execute main program
    '''
    # Grab user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--length',
        help='define binary string length (40 or -1)')
    args = parser.parse_args()
    if args.length == '50':
        print("Generating binary strings of length 40")
        main(length=50)
    elif args.length == '-1':
        print("Generating binary strings of length b/w 1 and 40")
        main(length=-1)
    else:
        print('Invalid entry')
