import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.features.constants import train_dataset, test_dataset, val_dataset, model_saving, number_classes, batch_size
from src.features.functions import glasses_recognition, creating_generators, model_history

# Using GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    epochs = 100
    learning_rate = 0.0001
    train_generator, val_generator, test_generator = creating_generators(batch_size, train_dataset, test_dataset, val_dataset)

    model_saver = ModelCheckpoint(
        model_saving,
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    early_stop = EarlyStopping(monitor="val_accuracy", patience=20)
    model = glasses_recognition(number_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=val_generator,
                  callbacks=[model_saver, early_stop]
                  )
    model_history(history)
    print("Model training has been finished!")

if __name__ == "__main__":
    main()
