import tensorflow as tf
from keras.layers import Conv2D, Dropout, Dense, BatchNormalization, MaxPool2D,Add,GlobalAveragePooling2D

class ResnetBlock(tf.keras.Model):
    def __init__(self,filter_size,downSample=False):
        super(ResnetBlock,self).__init__()
        self.filter = filter_size
        self.downSample = downSample
        stride = 2 if downSample==True else 1
        
        self.conv1 = Conv2D(self.filter,(3,3), (stride,stride), activation="relu",padding="same")
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(self.filter,(3,3), strides=(1,1), activation="relu",padding="same")
        self.batch_norm2 = BatchNormalization()
        self.merger = Add()
        
        if downSample == True:
            self.conv3 = Conv2D(self.filter,(1,1),(2,2),activation="relu",padding="same")
            self.batch_norm3 = BatchNormalization()

    def call(self,inputs):
        #inputs = tf.cast(inputs, dtype="float32")
        output = self.conv1(inputs)
        output = self.batch_norm1(output)
        x = tf.nn.relu(output)
        output = self.conv2(output)
        output = self.batch_norm2(output)
        
        if self.downSample == True:
            residual = self.conv3(inputs)
            residual = self.batch_norm3(residual)
            residual = self.merger([output,residual])
            x = tf.nn.relu(x)
            return residual
        
        else:
            output = self.merger([output,inputs])
            x = tf.nn.relu(x)
            return output
    

class ResidualNetwork18(tf.keras.Model):
    def __init__(self,classes):
        super(ResidualNetwork18,self).__init__()
        self.classes = classes
        self.initial_conv = Conv2D(64,(7,7),(2,2),activation="relu",padding="same")
        self.init_bn = BatchNormalization()
        self.pool = MaxPool2D((2, 2), strides=2, padding="same")
        self.resnetBlock1_1 = ResnetBlock(64)
        self.resnetBlock1_2 = ResnetBlock(64)
        self.resnetBlock2_1 = ResnetBlock(128,downSample=True)
        self.resnetBlock2_2 = ResnetBlock(128)
        self.resnetBlock3_1 = ResnetBlock(256,downSample=True)
        self.resnetBlock3_2 = ResnetBlock(256)
        self.resnetBlock4_1 = ResnetBlock(512,downSample=True)
        self.resnetBlock4_2 = ResnetBlock(512)
        self.pool_1 = GlobalAveragePooling2D()
        #self.dropout = Dropout(0.5)
        self.dense = Dense(self.classes,activation="softmax")
        
    def call(self,inputs):
        
        output = self.initial_conv(inputs)
        #print(output.shape)
        output = self.init_bn(output)
        #print(output.shape)
        output = self.pool(output)
        #print(output.shape)
        output = self.resnetBlock1_1(output)
        #print(output.shape)
        output = self.resnetBlock1_2(output)
        #print(output.shape)
        output = self.resnetBlock2_1(output)
        #print(output.shape)
        output = self.resnetBlock2_2(output)
        #print(output.shape)
        output = self.resnetBlock3_1(output)
        #print(output.shape)
        output = self.resnetBlock3_2(output)
        #print(output.shape)
        output = self.resnetBlock4_1(output)
        #print(output.shape)
        output = self.resnetBlock4_2(output)
        #print(output.shape)
        output = self.pool_1(output)
        #print(output.shape)
        output = self.dense(output)#self.dropout(output))
        #print(output.shape)
        
        return output
    
    
if __name__ == "__main__":
    import numpy as np
    
    Resnet = ResidualNetwork18(7)
    Resnet.call(np.random.random(size=(1,224,224,3)))