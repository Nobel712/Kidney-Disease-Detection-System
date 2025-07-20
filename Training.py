train=""
val=""
data_iterator = train.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i in range(4):  
    for j in range(4):  
        index = i * 4 + j  
        ax[i, j].imshow(batch[0][index].astype(int))
        ax[i, j].set_title(label_to_class_name[batch[1][index]])
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

model=""
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',  # or 'categorical_crossentropy' for one-hot labels
    metrics=['accuracy']
)

import time
from tensorflow.keras.callbacks import ModelCheckpoint

# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_model_weights.h5',    # or use .keras for full model format
    save_weights_only=True,              # Set False to save the full model
    save_best_only=True,                 # Only save when val loss improves
    monitor='loss',                      # or 'val_loss' if you have validation
    mode='min',
    verbose=1
)

epochs =   #"10 -20"

start_time = time.time()

history = model.fit(
     train,
     validation_data=val,
    batch_size=16,
    epochs=epochs,
  callbacks=[
         ..........      # your custom callback
        checkpoint_callback # model saving callback
    ]
)

end_time = time.time()
training_time = end_time - start_time

print(f"Total training time: {training_time:.2f} seconds")


