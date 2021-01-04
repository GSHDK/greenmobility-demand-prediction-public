import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.ConvLSTM import ConvLSTMCell

class VanillaConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(VanillaConvLSTM, self).__init__()

        """ ARCHITECTURE 
        GridData
        ⬇️
        ConvLSTM
        ⬇️
        ConvLSTM
        ⬇️
        3D CNN
        ⬇️
        Prediction
        
        
        
        # Encoder (ConvLSTM)

        """
        self.convlstm1 = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convlstm2 = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)


        self.output_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def encoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.convlstm1(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t]) 
            h_t2, c_t2 = self.convlstm2(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2]) 
        
        outputs.append(h_t2)
        
        if future_step>1:
            raise NotImplementedError('Many predictions')

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.output_CNN(outputs)
        outputs = torch.nn.ReLU()(outputs)
        return outputs
        
        
    def forward(self, x, future_seq=1, hidden_state=None, cuda=False):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # Set cuda
        if cuda and torch.cuda.is_available():
            x = x.cuda()

        # initialize hidden states
        h_t, c_t = self.convlstm1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.convlstm2.init_hidden(batch_size=b, image_size=(h, w))
        
        # autoencoder forward
        outputs = self.encoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2)

        return outputs


# CLSTM
class VanillaConvLSTMFC(nn.Module):
    def __init__(self, nf, in_chan, grid_shape=(5,5)):
        super(VanillaConvLSTMFC, self).__init__()

        """ ARCHITECTURE 
        GridData
        ⬇️
        ConvLSTM
        ⬇️
        ConvLSTM
        ⬇️
        3D CNN
        ⬇️
        FC
        ⬇️
        Prediction

        """
        self.convlstm1 = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convlstm2 = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.grid_shape = grid_shape

        self.output_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        # image size + featues in, x out
        self.FC_CNN = nn.Linear(grid_shape[0]*grid_shape[1], grid_shape[0]*grid_shape[1])
        # x in , image size out
        
        self.activation = nn.ReLU()
        


    def encoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.convlstm1(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t]) 
            h_t2, c_t2 = self.convlstm2(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2]) 
        
        outputs.append(h_t2)
        
        if future_step>1:
            raise NotImplementedError('Many predictions')

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.output_CNN(outputs)
        outputs = torch.nn.ReLU()(outputs)
        return outputs
        
        

    def forward(self, x, future_seq=1, hidden_state=None, cuda=False):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # Set cuda
        if cuda and torch.cuda.is_available():
            x = x.cuda()

        # initialize hidden states
        h_t, c_t = self.convlstm1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.convlstm2.init_hidden(batch_size=b, image_size=(h, w))
        
        # autoencoder forward
        outputs = self.encoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2)
        # First CNN
        outputs = self.FC_CNN(outputs.view(-1))
        # Second CNN
        #outputs = self.FC_OUT(outputs)

        return self.activation(outputs.view(1,1,1,self.grid_shape[0],-1))



# Encoder -> Forecaster MNIST
class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None, cuda=False):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # Set cuda
        if cuda and torch.cuda.is_available():
            x = x.cuda()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w), cuda=cuda)
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w), cuda=cuda)
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w), cuda=cuda)
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w), cuda=cuda)

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs



# CLSTM-BN-F
class VanillaConvLSTMFCNormFeatures(nn.Module):
    def __init__(self, nf, in_chan,features_fc, grid_shape=(5,5)):
        super(VanillaConvLSTMFCNormFeatures, self).__init__()

        """ ARCHITECTURE 
        GridData
        ⬇️
        ConvLSTM
        ⬇️
        ConvLSTM
        ⬇️
        3D CNN
        ⬇️
        FC
        ⬇️
        Prediction

        Best results config:
        NF=30
        Features: Day of week, Day of month, time of day
        """
        self.convlstm1 = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convlstm2 = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.grid_shape = grid_shape

        self.output_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        
        # Keep gridsize to make it more generix
        self.grid_shape = grid_shape
        grd = grid_shape[0]*grid_shape[1]
        # image size + featues in, x out
        self.FC_CNN = nn.Linear(grd+features_fc,grd)
        # x in , image size out
        self.FC_OUT = nn.Linear(grd,grd)
        


    def encoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.convlstm1(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t]) 
            h_t2, c_t2 = self.convlstm2(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2]) 
        
        outputs.append(h_t2)
        
        if future_step>1:
            raise NotImplementedError('Many predictions')

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.output_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)
        return outputs
        
        

    def forward(self, x, y, future_seq=1, hidden_state=None, cuda=False):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # Set cuda
        if cuda and torch.cuda.is_available():
            x = x.cuda()

        # initialize hidden states
        h_t, c_t = self.convlstm1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.convlstm2.init_hidden(batch_size=b, image_size=(h, w))
        
        # autoencoder forward
        outputs = self.encoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2)

        # Add Features
        feature_cat_output = torch.cat((outputs.view(-1),y.view(-1)))
        
        # CNN to FC
        outputs = self.FC_CNN(feature_cat_output)
        
        # Activation functions
        #outputs = self.activation(outputs)
        
        
        # Second CNN -> output
        outputs = self.FC_OUT(outputs.view(-1))

        return outputs.view(1,1,1,self.grid_shape[0],-1)


# CLSTM-BN
class VanillaConvLSTMFCNorm(nn.Module):
    def __init__(self, nf, in_chan, grid_shape=(5,5)):
        super(VanillaConvLSTMFCNorm, self).__init__()

        """ ARCHITECTURE 
        GridData
        ⬇️
        ConvLSTM
        ⬇️
        ConvLSTM
        ⬇️
        3D CNN
        ⬇️
        FC
        ⬇️
        Prediction

        """
        self.convlstm1 = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convlstm2 = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.grid_shape = grid_shape

        self.output_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        # image size + featues in, x out
        self.FC_CNN = nn.Linear(grid_shape[0]*grid_shape[1], grid_shape[0]*grid_shape[1])
        


    def encoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.convlstm1(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t]) 
            h_t2, c_t2 = self.convlstm2(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2]) 
        
        outputs.append(h_t2)
        
        if future_step>1:
            raise NotImplementedError('Many predictions')

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.output_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)
        return outputs
        
        

    def forward(self, x, future_seq=1, hidden_state=None, cuda=False):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # Set cuda
        if cuda and torch.cuda.is_available():
            x = x.cuda()

        # initialize hidden states
        h_t, c_t = self.convlstm1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.convlstm2.init_hidden(batch_size=b, image_size=(h, w))
        
        # autoencoder forward
        outputs = self.encoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2)
        # First CNN
        outputs = self.FC_CNN(outputs.view(-1))
        # Second CNN
        #outputs = self.FC_OUT(outputs)

        return outputs.view(1,1,1,self.grid_shape[0],-1)



# LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim , batch_first=False)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1,1, self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(1,1, self.hidden_dim).requires_grad_()

  
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step

        out = self.fc(out[:, -1, :]) 
        return out
