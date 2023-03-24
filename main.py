
'''
author: Roberto Scalas 
date:   2023-03-24 10:53:03.479285
'''

from streamlit_drawable_canvas import st_canvas
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchviz import make_dot
from questions_answers import questions


# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                                    train=False,                
                                    transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,  
                                            shuffle=False)

for i, (images, labels) in enumerate(train_loader):
    print(images.shape)
    print(labels.shape)
    break

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x, show_conv = False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        if show_conv:
            conv_1_ = make_subplots(rows=1, cols=10, subplot_titles=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10') )
            for i in range(10):
                conv_1_.add_trace(go.Heatmap(z=x[0,i,:,:].detach().numpy(), showscale=False), row=1, col=i+1,) 
                # make the graph smaller
                conv_1_.update_layout(height=250, width=500)
                # no legend
                conv_1_.update_layout(showlegend=False)
                # no axis
                conv_1_.update_xaxes(showticklabels=False)
                conv_1_.update_yaxes(showticklabels=False)
                
            conv_1_.update_layout(title_text=f'Convolution 1 : Max pooling : ReLu -> SHAPE: {x.shape}')

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        if show_conv:
            conv_2_ = make_subplots(rows=1, cols=20, subplot_titles=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20') )
            for i in range(20):
                conv_2_.add_trace(go.Heatmap
                (z=x[0,i,:,:].detach().numpy(), showscale=False), row=1, col=i+1)
                # make the graph smaller
                conv_2_.update_layout(height=250, width=500)
                # set title 
                conv_2_.update_layout(title_text=f'Convolution 2 : Max pooling : ReLu -> SHAPE: {x.shape}')
                # no axis
                conv_2_.update_xaxes(showticklabels=False)
                conv_2_.update_yaxes(showticklabels=False)
                # no lines in the graph
                conv_2_.update_xaxes(showline=False, linewidth=0, linecolor='black')
        
        x = x.view(-1, 320)
        
        if show_conv:
            flatten_ = go.Figure(data=go.Heatmap(z=x.detach().numpy(),showscale=False))
            flatten_.update_layout(title_text=f'Mono-dimensional -> SHAPE: {x.shape}')
            # make the graph smaller
            flatten_.update_layout(height=250, width=500)

        x = F.relu(self.fc1(x))
        
        if show_conv:
            ff1_ = go.Figure(data=go.Heatmap(z=x.detach().numpy(), showscale=False))
            ff1_.update_layout(title_text=f'Fully connected 1 : ReLu -> SHAPE: {x.shape}')
            # make the graph smaller
            ff1_.update_layout(height=250, width=500)

        x = F.dropout(x, training=self.training)
        
        if show_conv:
            drop_ = go.Figure(data=go.Heatmap(z=x.detach().numpy(), showscale=False))
            drop_.update_layout(title_text=f'Dropout -> SHAPE: {x.shape}')
            # make the graph smaller
            drop_.update_layout(height=250, width=500)

        x = self.fc2(x)
        if show_conv:
            ff2 = go.Figure(data=go.Heatmap(z=x.detach().numpy(), showscale=False))
            ff2.update_layout(title_text=f'Fully connected 2 -> SHAPE: {x.shape}')
            # make the graph smaller
            ff2.update_layout(height=250, width=500)
        
        x = F.log_softmax(x, dim=1) #
        
        if show_conv:
            out = go.Figure(data=go.Heatmap(z=x.detach().numpy(), showscale=False))
            out.update_layout(title_text=f'log_softmax -> SHAPE: {x.shape}')
            # make the graph smaller
            out.update_layout(height=250, width=500)
        
        if show_conv:
            st.plotly_chart(conv_1_, use_container_width=True)
            st.plotly_chart(conv_2_, use_container_width=True)
            stacked = make_subplots(rows=5, cols=1, subplot_titles=('Flatten', 'Fully connected 1', 'Dropout', 'Fully connected 2', 'log_softmax') )
            stacked.add_trace(go.Heatmap(z=flatten_.data[0]['z'], showscale=False), row=1, col=1)
            stacked.add_trace(go.Heatmap(z=ff1_.data[0]['z'], showscale=False), row=2, col=1)
            stacked.add_trace(go.Heatmap(z=drop_.data[0]['z'], showscale=False), row=3, col=1)
            stacked.add_trace(go.Heatmap(z=ff2.data[0]['z'], showscale=False), row=4, col=1)
            stacked.add_trace(go.Heatmap(z=out.data[0]['z'], showscale=False), row=5, col=1)
            # show
            st.plotly_chart(stacked, use_container_width=True)

        return x

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    dataframe = pd.DataFrame()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data.item())
        st.write(message)
        # add the loss to the dataframe
        dataframe = pd.concat([dataframe, pd.DataFrame({'epoch': epoch, 'batch_idx': batch_idx, 'loss': loss.data.item()}, index=[0])], ignore_index=True)

    return dataframe

def test(epoch):
    dataframe_test_loss = pd.DataFrame()
    model.eval()
    test_loss = 0
    correct = 0
    # use no_grad() to avoid tracking history
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    message = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    accuracy = 100. * correct / len(test_loader.dataset)
    if accuracy > 85:
        st.sidebar.success(message)
    elif accuracy > 70 and accuracy < 85:
        st.sidebar.info(message)
    else:
        st.sidebar.error(message)

    # add the loss to the dataframe only loss and epoch
    dataframe_test_loss = pd.concat([dataframe_test_loss, pd.DataFrame({'epoch': epoch, 'loss': test_loss}, index=[0])], ignore_index=True)


    return dataframe_test_loss

def train_for_n_epochs(n):
    print('Starting training for {} epochs'.format(n))
    data_frame_losses = pd.DataFrame()
    dataframe_loss_test = pd.DataFrame()
    with st.sidebar.expander('Train the model'):
        for epoch in range(1, n + 1):
            dataframe_loss = train(epoch)
            # concatenate the dataframes
            data_frame_losses = pd.concat([data_frame_losses, dataframe_loss])
            dataframe_test_loss = test(epoch)
            # add the test loss to the dataframe
            dataframe_loss_test = pd.concat([dataframe_loss_test, dataframe_test_loss])
    st.success('Finished training: Saved PyTorch Model State to conv_mnist_ffn.pt')
    torch.save(model.state_dict(), 'conv_mnist_ffn.pt')

    # reset the index
    data_frame_losses = data_frame_losses.reset_index(drop=True)
    dataframe_loss_test = dataframe_loss_test.reset_index(drop=True)

    import plotly.graph_objects as go
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=data_frame_losses.index, y=data_frame_losses['loss'], mode='lines', name='train'))
    fig_train.update_layout(title='Train loss', xaxis_title='Batch', yaxis_title='Loss')
    st.plotly_chart(fig_train, use_container_width=True)

    return data_frame_losses, dataframe_test_loss

def try_model_predict(sample):
    # load the model
    model = ConvNet()
    model.load_state_dict(torch.load('conv_mnist_ffn.pt'))
    model.eval()

    # convert the sample to a tensor
    sample = torch.from_numpy(sample).float()
    # get the prediction
    output = model.forward(sample, show_conv=True)
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return classes[pred.item()]

start_model = st.sidebar.button('Train the model')
epochs = st.sidebar.number_input('Number of epochs', min_value=1, max_value=100, value=10)

if start_model:
    train_for_n_epochs(epochs)

#------------------------------------------- Streamlit -------------------------------------------#
st.title('Convolutional Neural Network')
expander_try_it = st.expander('Draw a digit')
expander_training_data = st.expander('Training Data', expanded=True)
expander_architecture = st.expander('Model Architecture')
expander_dataset = st.expander('Process Data')
expander_model = st.expander('Model')
expander_train = st.expander('Training')
expander_test = st.expander('Testing')
expander_functions = st.expander('Usage')

# -------------------- #    Streamlit    # -------------------- #

expander_dataset.write(
'''
This is a convolutional neural network that can classify handwritten digits.
The model was trained on the MNIST dataset.
```python
# create a mnist convolutional network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import streamlit as st

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                                    train=False,                
                                    transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,  
                                            shuffle=False)

# take a look at the data
for i, (images, labels) in enumerate(train_loader):
    print(images.shape)
    print(labels.shape)
    break
```''')

expander_model.write('''
```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```''')

expander_train.write('''
```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
```''')

expander_test.write('''
```python
def test():
    model.eval()
    test_loss = 0
    correct = 0
    # use no_grad() to avoid tracking history
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
''')

expander_functions.write('''
```python      
def train_for_n_epochs(n):
    try:
        # load the model
        model = ConvNet()
        model.load_state_dict(torch.load('conv_mnist_ffn.pt'))
    except:
        model = ConvNet()

    with st.expander('Training'):
        for epoch in range(1, n):
            train(epoch)
            test()
            # save the model
            torch.save(model.state_dict(), 'conv_mnist_ffn.pt')

def try_model_predict(sample):
    # load the model
    model = ConvNet()
    model.load_state_dict(torch.load('conv_mnist_ffn.pt'))
    model.eval()

    # convert the sample to a tensor
    sample = torch.from_numpy(sample).float()
    # get the prediction
    output = model(sample)
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return classes[pred.item()]
```
''')


with expander_training_data:
    c,d = st.columns(2)
    image_container = c.empty()
    image_sample = d.slider('Image sample', 0, 10000, 0, 1)
    sample = test_dataset[image_sample][0].numpy()
    sample = sample.reshape(1, 1, 28, 28)
    pred = try_model_predict(sample)
    image_container.image(sample.reshape(28, 28), caption='Prediction: ' + pred, width=200)

with expander_try_it:
    c1,c2 = st.columns(2)

    with c2:
      canvas_result = st_canvas(
         fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
         stroke_width=15,
         stroke_color='black',
         background_color='white',
         background_image=None,
         update_streamlit=True,
         height=150,
         drawing_mode='freedraw',
         point_display_radius=0,
         key="canvas",
      )
      # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        # it need to be non-empty
        # transform to numpy array 28x28
        image = Image.fromarray(canvas_result.image_data)
        image = image.resize((28,28))
        image = image.convert('L')
        image = np.array(image)
        image = image.reshape(1,1,28,28)

        container = st.container()
        prediction = try_model_predict(image)
        container.info(f'The predicted class is: {prediction}')

with expander_architecture:
    c1, c2 = st.columns(2)
    # visualize the model
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    # architecture or parameters
    g = make_dot(y, params=dict(model.named_parameters()))
    c1.write(g)

    
    selected_question = c2.selectbox('Select a question', list(questions.keys()))
    # random question
    
    c2.write(questions[selected_question])