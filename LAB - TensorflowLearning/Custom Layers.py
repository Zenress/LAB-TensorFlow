import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

#In the tf.keras.layers package, layers are objects. To construct a layer,
#simply construct the object. Most layers take as a first argument the number
#of output dimension / channels.
layer = tf.keras.layers.Dense(100)
#The number of input dimensions is often unnecessary, as it can be inferred
#the first time the layer is used, but it can be provided if you want to
#specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None,5))

#To use a layer, simply call it.
layer(tf.zeros([10,5]))

#Layers have many useful methods. For example, you can inspect all variables
#In a layer using 'layer.variables' and trainable variables using
#'layer.trainable_variables'. In this case a fully-connected layer
#will have variables for weights and biases.
print(layer.variables)

#The variables are also accessible through nice accessors
print(layer.kernel, layer.bias)

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
      super(MyDenseLayer, self).__init__()
      self.num_outputs = num_outputs
      
  def build(self,input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    
  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)
  
layer = MyDenseLayer(10)


_ = layer(tf.zeros([10, 5])) #Calling the layer '.builds' it.
print([var.name for var in layer.trainable_variables])


class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
      super(ResnetIdentityBlock, self).__init__(name='')
      filters1, filters2, filters3 = filters
      
      self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
      self.bn2a = tf.keras.layers.BatchNormalization()
      
      self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
      self.bn2b = tf.keras.layers.BatchNormalization()
      
      self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
      self.bn2c = tf.keras.layers.BatchNormalization()
      
  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)
    
    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)
    
    x = self.conv2c(x)
    x = self.bn2c(x, training=training)
    
    x += input_tensor
    return tf.nn.relu(x)
  
block = ResnetIdentityBlock(1,[1,2,3])

_ = block(tf.zeros([1,2,3,3]))
print(block.layers)
len(block.variables)

block.summary()

my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1,(1,1),
                                                     input_shape=(
                                                       None, None, 3)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2,1, padding='same'),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3,(1,1)),
                              tf.keras.layers.BatchNormalization()])
print(my_seq(tf.zeros([1,2,3,3])))

my_seq.summary()