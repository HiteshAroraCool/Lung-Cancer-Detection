###Data Genrator¶
def (train, test, all_labels):

img_size = (128, 128)

datagen = ImageDataGenerator(
    rotation_range=40,          # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,      # Shift images horizontally (fraction of total width)
    height_shift_range=0.2,     # Shift images vertically (fraction of total height)
    zoom_range=0.2,             # Zoom in on images
    horizontal_flip=True,       # Flip images horizontally
    fill_mode='nearest',        # Strategy for filling in new pixels (after a transformation)
)
train['newLabel'] = train.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
test['newLabel'] = test.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

train_gen = datagen.flow_from_dataframe(
    dataframe=train,
    directory=None,
    x_col = 'image path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    classes = all_labels,
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 32)

valid_gen = datagen.flow_from_dataframe(
    dataframe=test,
    directory=None,
    x_col = 'image path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    classes = all_labels,
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 32)

test_x, test_y = next(datagen.flow_from_dataframe(
    test,
    directory=None,
    x_col='image path',
    y_col='newLabel',
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 1024
))

return train_gen, valid_gen, test_x, test_y, img_size