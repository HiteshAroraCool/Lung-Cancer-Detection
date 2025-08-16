
def (train_gen, valid_gen, img_size)
###Loss Function: I will use sigmoid activation function, we convert the multi-label problem into multiple binary classification problems, where each class is predicted independently, and we apply binary crossentropy loss to optimize the model's performance.

previour_model_weights = "weight_file"
epochs = 5
n_class = 14

checkpoint = ModelCheckpoint(
    filepath='best_model.weights.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
    initial_value_threshold=None
)

earlystopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

base_model = MobileNet(input_shape = (img_size[0], img_size[1], 1), include_top = False, weights = previour_model_weights)

#Tranfer Learning Model
model = Sequential([

    # Base Layer
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),

    # Head Layer
    Flatten(),
    
    # FC Layer-1
    Dense(140, activation='relu'),
    Dropout(0.3),

    # FC Layer-2 (output Layer)
    Dense(n_class, activation='sigmoid')
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
history = model.fit(
    train_gen,                        # Training data generator
    validation_data=valid_gen,         # Validation data generator
    epochs=epochs,                         # Number of epochs
    callbacks=[earlystopping, checkpoint],               # List of callbacks
    steps_per_epoch=100,
)