import onlinehd
import numpy as np
from time import time
import pandas as pd
from onlinehd import Encoder


class HDFusion:
    def __init__(self, classes, features_1, features_2, dim, device=None):
        self.classes = classes
        self.features_1 = features_1
        self.features_2 = features_2
        self.encoder_1 = Encoder(features_1, dim)
        self.encoder_2 = Encoder(features_2, dim)
        self.dim = dim
        self.model = onlinehd.OnlineHD(classes, features_1, dim)
        self.init = False
        self.df = pd.DataFrame(columns=['X1', 'X2', 'Confidence', 'Accuracy', 'Alpha'])

        if device == 'cuda':
            print("Using CUDA GPU")
            self.device = 'cuda:2'
        elif device == 'mps':
            print("Using M1 GPU")
            self.device = 'mps'
        else:
            print("Using CPU")
            self.device = 'cpu'
        self.model.to(self.device)
        
    def encode(self, bool, data):
        if bool:
            return self.encoder_1(data)
        else:
            return self.encoder_2(data)

    def set_data(self, X1, Y1, X2, Y2):
        encoded_X1 = self.encode(True, X1)
        encoded_X2 = self.encode(False, X2)
        self.X1 = encoded_X1.to(self.device)
        self.X2 = encoded_X2.to(self.device)
        for y1, y2 in zip(Y1, Y2):
            if(y1 != y2):
                print("ERROR WITH DATASET, DIFFERENT LABELS")
                self.Y = Y1.to(self.device)
            else:
                self.Y = Y1.to(self.device)
    
    def register_accuracies(self, X1, X2, confidence, accs, alphas):
        for x1, x2, conf, acc, a in zip(X1, X2, confidence, accs, alphas):
            dict_ = {'X1':x1, 'X2':x2, 'Confidence':conf, 'Accuracy':acc, 'Alpha':a}
            self.df = self.df.append(dict_, ignore_index = True)


    def train(self, epochs=30):
        alpha = 0.5
        print(f"=======alpha={alpha}========")
        print("Training...")
        X = [(alpha*x1+(1-alpha)*x2) for x1, x2 in zip(self.X1, self.X2)]
        X = X.to(self.device)
        Y = self.Y
        print("Training set shape", X.shape)
        t = time()
        self.model = self.model.fit(X, Y, bootstrap=1.0, lr=0.035, epochs=epochs, encoded=True)
        training_time = time() - t
        print("Training time :", training_time)
        Yhat = self.model(X)
        acc = (Y == Yhat).float().mean()
        print("Training Accuracy", acc)
    
    def train_fusion(self, epochs=5, alpha=0.5):
        if not self.init:
            self.train()
        print(f"=======alpha={alpha}========")
        print("Retraining...")
        X = [(alpha*x1+(1-alpha)*x2) for x1, x2 in zip(self.X1, self.X2)]
        X = X.to(self.device)
        Y = self.Y
        print("Training set shape", X.shape)
        t = time()
        self.model = self.model.fit(X, Y, bootstrap=1.0, lr=0.035, epochs=epochs, encoded=True)
        training_time = time() - t
        print("Training time :", training_time)
        Yhat = self.model(X)
        acc = (Y == Yhat).float().mean()
        print("Retraining Accuracy", acc)

    def test_inference(self, X1, Y1, X2, Y2, alpha):
        encoded_X1 = self.encode(True, X1)
        encoded_X2 = self.encode(False, X2)
        for y1, y2 in zip(Y1, Y2):
            if(y1 != y2):
                print("ERROR WITH DATASET, DIFFERENT LABELS")
            else:
                Y = Y1.to(self.device)
        
        X = [(alpha*x1+(1-alpha)*x2) for x1, x2 in zip(encoded_X1, encoded_X2)]
        X = X.to(self.device)
        # TODO: Implement confidence function
        # confidences = [self.model.get_confidence(x) for x in X]
        print("Testing set shape", X.shape)
        t = time()
        Y_hat = self.model(X, encoded=True)
        inference_time = time() - t
        print("Inference time :", inference_time)
        acc = (Y == Y_hat).float().mean()
        print("Test Accuracy", acc)
        accs = np.ones(self.dim) * acc
        alphas = np.ones(self.dim) * alpha
        self.register_accuracies(encoded_X1, encoded_X2, confidences, accs, alphas)
        return acc, encoded_X1, encoded_X2

    def filter_df(self, filename, encoded_X1, encoded_X2):
        if filename == None:
            filename= './results/fusion_results_{}.csv'.format(self.dim)
        # df = pd.DataFrame()
        ids = []
        for x1, x2 in zip(encoded_X1, encoded_X2):
            df2 = self.df.loc[(df['X1'] == x1) & (df['X2'] == x2)]
            ids.append(df2['Similarity'].idxmax())
        
        df = sef.df.iloc[ids]
        df = df.reset_index(drop=True)
        df.to_csv(filename, index=False)

