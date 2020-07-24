# classification.py


import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


pd.set_option('display.max_columns', None)


LANG = 'Lang'
CHART = 'Chart'
CLOSURE = 'Closure'
MOCKITO = 'Mockito'
MATH = 'Math'
TIME = 'Time'
PAPER_SUBJECTS = {
    CLOSURE: [1, 5, 8, 9, 12, 13, 14, 15, 28, 36, 46, 49, 55, 58, 67, 72, 88, 91, 92, 98, 102, 108, 111, 116, 124, 130,
              132],
    LANG: [1, 2, 4, 11, 25, 28, 33, 37, 40, 44, 45, 49, 51, 53, 54, 55, 59, 62],
    MATH: [10, 2, 4, 7, 21, 22, 25, 26, 28, 33, 35, 39, 40, 51, 52, 57, 60, 68, 69, 72, 76, 79, 82, 84, 86, 89, 95, 96,
           100, 103, 105],
}


class MutantUtilityLSTM(object):

    def __init__(self,
                 cmd_path='../data/customized-mutants.csv'):
        self._cmd_path = cmd_path

    def train_model(self):
        y_pred = []
        y_true = []
        # Generate training data from AST
        df, X, y = self._generate_training_data()
        # Leave-one-project-out cross-validation (i.e. predict one project and train on the others)
        for project in PAPER_SUBJECTS.keys():
            print('--------------------------------------------------------')
            print('Predicting equivalence for {} project...'.format(project))
            print('--------------------------------------------------------\n')
            trn_i = df.index[df['projectId'] != project].values
            tst_i = df.index[df['projectId'] == project].values
            trn_x = X[trn_i]
            trn_y = y[trn_i]
            tst_x = X[tst_i]
            tst_y = y[tst_i]
            # Build LSTM model
            model = Sequential()
            model.add(Embedding(8000, 64, input_length=X.shape[1]))
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(64))
            model.add(Dense(1, activation='sigmoid'))
            optimizer = Adam(learning_rate=0.00007)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
            # Train model
            model.fit(trn_x, trn_y, epochs=15, batch_size=64,
                      callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.0001)])
            # Evaluate validation set
            # auc = model.evaluate(tst_x, tst_y)
            # print('Loss: {:0.3f}\t\tAUC: {:0.3f}'.format(auc[0], auc[1]))
            # Predict validation set
            pred = model.predict(tst_x)
            y_pred.append(pred)
            y_true.append(tst_y)
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        # Compute overall AUC for all projects
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        print('Overall AUC: {}'.format(roc_auc))

    def _generate_training_data(self):
        df = pd.read_csv(self._cmd_path)
        df['ast_text'] = df.apply(self._ast_to_text, axis=1)
        # Tokenize AST nodes as text
        tokenizer = Tokenizer(filters='!"#$%&,./?@[\]^`{|}~', lower=True)
        tokenizer.fit_on_texts(df['ast_text'].values)
        X = tokenizer.texts_to_sequences(df['ast_text'].values)
        X = pad_sequences(X)
        y = df['isKilled'].values
        y = y.reshape(y.shape[0], 1)
        return df, X, y

    @staticmethod
    def _ast_to_text(row):
        # MUTATIONOPERATOR, PARENTSTMTCONTEXTDETAILED, NODETYPEBASIC, HASVARIABLE, HASLITERAL, HASOPERATOR
        try:
            text = ' '.join([row['parentStmtContextDetailed'], row['astContextDetailed'], row['nodeTypeBasic'],
                             row['mutationOperator']])
        except:
            text = ''
        return text


if __name__ == '__main__':
    mu = MutantUtilityLSTM()
    mu.train_model()
