import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

print(x_train.shape)
print(y_train.shape)

y_train = tf.squeeze(tf.one_hot(y_train, 100), axis=1)
y_test = tf.squeeze(tf.one_hot(y_test, 100), axis=1)

print(y_train.shape)
print(y_test.shape)

base_model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3))
base_model = tf.keras.models.Model(base_model.inputs, base_model.layers[-2].output)
x = base_model.output
pred = tf.keras.layers.Dense(100, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs = pred)

opt= tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss = 'categorical_crossentropy', metrics=['acc'])

model.summary()

history = model.fit(x=x_train, y= y_train, batch_size=32,
                    epochs=20, validation_data=(x_test, y_test))

print('validation accuracy')
print(history.history['val_acc'][-1])

results = model.evaluate(x_test, y_test, batch_size=32)
print('test accuracy')
print(results[1])