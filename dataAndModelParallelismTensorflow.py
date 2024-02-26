import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, -1) / 255.0
X_train = np.repeat(X_train, 3, axis=-1)
X_train = tf.image.resize(X_train, [64, 64]).numpy()
y_train = to_categorical(y_train, 10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Setup for data parallelism
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Model part 1 (placed on a specific GPU for model parallelism)
    inputs1 = layers.Input(shape=(64, 64, 3))
    x1 = layers.Conv2D(32, 3, activation='relu')(inputs1)
    x1 = layers.MaxPooling2D()(x1)
    model_part1 = models.Model(inputs=inputs1, outputs=x1)

    # Model part 2 (could be placed on another GPU but within the same strategy scope for simplicity)
    inputs2 = layers.Input(shape=model_part1.output.shape[1:])
    x2 = layers.Conv2D(64, 3, activation='relu')(inputs2)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(64, activation='relu')(x2)
    outputs = layers.Dense(10, activation='softmax')(x2)
    model_part2 = models.Model(inputs=inputs2, outputs=outputs)

batch_size = 64
epochs = 5

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Distribute datasets
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

# Define loss and metrics
with strategy.scope():
    loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    val_accuracy = tf.keras.metrics.CategoricalAccuracy()

    optimizer = tf.keras.optimizers.Adam()

# Training step
def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        part1_output = model_part1(images, training=True)
        predictions = model_part2(part1_output, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model_part1.trainable_variables + model_part2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_part1.trainable_variables + model_part2.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss

# Validation step
def val_step(inputs):
    images, labels = inputs
    part1_output = model_part1(images, training=False)
    predictions = model_part2(part1_output, training=False)
    t_loss = loss_object(labels, predictions)

    val_accuracy.update_state(labels, predictions)

# Distribute the training step
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Distribute the validation step
@tf.function
def distributed_val_step(dataset_inputs):
    return strategy.run(val_step, args=(dataset_inputs,))

# Training loop
for epoch in range(epochs):
    # Train
    total_loss = 0.0
    num_batches = 0
    train_accuracy.reset_states()
    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = total_loss / num_batches

    # Validate
    val_accuracy.reset_states()
    for x in val_dist_dataset:
        distributed_val_step(x)

    print(f'Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy.result() * 100}, Validation Accuracy: {val_accuracy.result() * 100}')