#Training the model and plotting the loss and accuracy curves
model.load_weights(UNET.h5)

model.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train, seg_train, epochs=11, batch_size=128, validation_data=(validate, seg_validate))

test_loss, test_acc = model.evaluate(test, seg_test)

"""#Displaying the curves of loss and accuracy"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()