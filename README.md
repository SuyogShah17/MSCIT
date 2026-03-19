Practical 1: Introduction to TensorFlow 

```python
# BLOCK 1: IMPORTS & SETUP
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

# BLOCK 2: TENSORS & OPERATIONS
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[5, 6], [7, 8]])

# BLOCK 3: EXECUTION
add_result = tf.add(tensor_a, tensor_b)
mul_result = tf.matmul(tensor_a, tensor_b)
print("Addition:\n", add_result.numpy())
print("Multiplication:\n", mul_result.numpy())
```

---Practical 2: Linear Regression 

```python
# BLOCK 1: DATA PREP
import tensorflow as tf
import numpy as np
# Equation: y = 2x - 1
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# BLOCK 2: MODEL ARCHITECTURE
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# BLOCK 3: COMPILE & TRAIN
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, y, epochs=50, verbose=0)
print("Prediction for x=10 (Should be ~19):", model.predict([10.0]))
```

---Practical 3: Convolutional Neural Networks (Classification) 

# BLOCK 1: DATA PREP
import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1) # Reshape for CNN

# BLOCK 2: MODEL ARCHITECTURE
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# BLOCK 3: COMPILE & TRAIN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=32)
```

---Practical 4: Image Segmentation

# BLOCK 1: DATA PREP (Using synthetic data for standalone execution)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
X_seg = np.random.rand(10, 64, 64, 3) # 10 dummy RGB images
y_seg = np.random.randint(0, 2, (10, 64, 64, 1)) # 10 dummy binary masks

# BLOCK 2: MODEL ARCHITECTURE (Mini U-Net)
inputs = Input((64, 64, 3))
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2, 2))(c1)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
u1 = UpSampling2D((2, 2))(c2)
concat = concatenate([u1, c1])
outputs = Conv2D(1, (1, 1), activation='sigmoid')(concat)

# BLOCK 3: COMPILE & TRAIN
model_seg = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_seg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_seg.fit(X_seg, y_seg, epochs=2)
```

---Practical 5: Image Captioning using LSTM 


# BLOCK 1: DATA PREP (Synthetic features and sequences)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, Dropout
# Assume images are processed through ResNet50 into 2048-dim vectors
img_features = np.random.rand(100, 2048) 
# Assume text is tokenized into max length of 34
text_seqs = np.random.randint(1, 5000, (100, 34)) 
next_words = np.random.randint(0, 5000, (100, 1)) # Target labels
next_words_categorical = tf.keras.utils.to_categorical(next_words, num_classes=5000)

# BLOCK 2: MODEL ARCHITECTURE
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(34,))
se1 = Embedding(5000, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(5000, activation='softmax')(decoder2)

# BLOCK 3: COMPILE & TRAIN
model_cap = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
model_cap.compile(loss='categorical_crossentropy', optimizer='adam')
model_cap.fit([img_features, text_seqs], next_words_categorical, epochs=2)
```

---Practical 6: Autoencoder for Real-World Data 

# BLOCK 1: DATA PREP (Using sklearn breast cancer dataset)
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
data = load_breast_cancer()
X_real = StandardScaler().fit_transform(data.data) # Shape (569, 30)
input_dim = X_real.shape[1]

# BLOCK 2: MODEL ARCHITECTURE
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(14, activation='relu')(input_layer)
encoded = tf.keras.layers.Dense(7, activation='relu')(encoded) # Bottleneck
decoded = tf.keras.layers.Dense(14, activation='relu')(encoded)
output_layer = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)

# BLOCK 3: COMPILE & TRAIN
autoencoder_real = tf.keras.Model(inputs=input_layer, outputs=output_layer)
autoencoder_real.compile(optimizer='adam', loss='mean_squared_error')
autoencoder_real.fit(X_real, X_real, epochs=10, batch_size=32)
```

---Practical 7: Character Recognition (RNN vs CNN)

# BLOCK 1: DATA PREP (Synthetic character sequences)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Conv1D, GlobalMaxPooling1D, Dense
# 1000 sequences of 50 characters, 10 possible classes
X_chars = np.random.randint(0, 100, (1000, 50)) 
y_chars = np.random.randint(0, 10, (1000,))

# BLOCK 2: RNN ARCHITECTURE
rnn_model = tf.keras.Sequential([
    Embedding(input_dim=100, output_dim=32, input_length=50),
    SimpleRNN(64),
    Dense(10, activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# BLOCK 3: CNN ARCHITECTURE & TRAINING
cnn_model = tf.keras.Sequential([
    Embedding(input_dim=100, output_dim=32, input_length=50),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training RNN...")
rnn_model.fit(X_chars, y_chars, epochs=2, batch_size=32)
print("Training CNN...")
cnn_model.fit(X_chars, y_chars, epochs=2, batch_size=32)
```

---Practical 8: Autoencoders using MNIST

# BLOCK 1: DATA PREP
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # Flatten to 784

# BLOCK 2: MODEL ARCHITECTURE
input_img = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(32, activation='relu')(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)

# BLOCK 3: COMPILE & TRAIN
autoencoder_mnist = tf.keras.Model(input_img, decoded)
autoencoder_mnist.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder_mnist.fit(x_train, x_train, epochs=5, batch_size=256)
```

---Practical 9: RNN for Google Stock Price

# BLOCK 1: DATA PREP 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
# NOTE: Replace 'pd.DataFrame' with pd.read_csv('Google_Stock_Price_Train.csv') in the exam.
# Creating synthetic sequence data simulating stock scaling (0 to 1)
X_stock = np.random.rand(100, 60, 1) # 100 samples, 60 timesteps, 1 feature
y_stock = np.random.rand(100, 1)

# BLOCK 2: MODEL ARCHITECTURE
stock_model = tf.keras.Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# BLOCK 3: COMPILE & TRAIN
stock_model.compile(optimizer='adam', loss='mean_squared_error')
stock_model.fit(X_stock, y_stock, epochs=5, batch_size=32)
```

---Practical 10: GANs for Image Generation

# BLOCK 1: DATA PREP
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
(X_train_gan, _), (_, _) = mnist.load_data()
X_train_gan = (X_train_gan.astype(np.float32) - 127.5) / 127.5 # Normalize to [-1, 1]
X_train_gan = X_train_gan.reshape(-1, 784)

# BLOCK 2: MODEL ARCHITECTURES (Gen & Disc)
generator = tf.keras.Sequential([
    Dense(256, activation='relu', input_dim=100),
    Dense(784, activation='tanh')
])

discriminator = tf.keras.Sequential([
    Dense(256, activation='relu', input_dim=784),
    Dense(1, activation='sigmoid')
])
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# BLOCK 3: COMBINED GAN ARCHITECTURE & TRAINING LOOP (1 Epoch Demo)
discriminator.trainable = False
gan_input = Input(shape=(100,))
gan = tf.keras.Model(gan_input, discriminator(generator(gan_input)))
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Minimal Training Step (Use this loop in the exam)
batch_size = 32
noise = np.random.normal(0, 1, size=[batch_size, 100])
fake_images = generator.predict(noise)
real_images = X_train_gan[np.random.randint(0, X_train_gan.shape[0], batch_size)]

X_combined = np.concatenate([real_images, fake_images])
y_combined = np.zeros(2 * batch_size)
y_combined[:batch_size] = 0.9 # Label smoothing for real

d_loss = discriminator.train_on_batch(X_combined, y_combined)
noise_labels = np.ones(batch_size) # Try to trick discriminator
g_loss = gan.train_on_batch(noise, noise_labels)
print(f"D Loss: {d_loss}, G Loss: {g_loss}")
```

