from sklearn.model_selection import train_test_split
import numpy as np
import PIL.Image as PImage
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def prepare_dataset(data_dir, IMG_SHAPE):
    flowers_images_dict = {
        'daisy': list(data_dir.glob('daisy/*')),
        'dandelion': list(data_dir.glob('dandelion/*')),
        'roses': list(data_dir.glob('roses/*')),
        'sunflowers': list(data_dir.glob('sunflowers/*')),
        'tulips': list(data_dir.glob('tulips/*')),
    }

    flowers_labels_dict = {

        'daisy': 0,
        'dandelion': 1,
        'roses': 2,
        'sunflowers': 3,
        'tulips': 4,
    }

    X, y = [], []

    for flower_name, images in flowers_images_dict.items():
        # print(f'\n{flower_name}')
        # print(len(images))
        for image in images:
            resized_img = PImage.open(image).resize((IMG_SHAPE, IMG_SHAPE))
            x1 = np.array(resized_img)
            X.append(x1)
            # the number for each flower: from flowers labels dict :
            y.append(flowers_labels_dict[flower_name])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    print("Train size:{}".format(len(X_train)))
    print("Test size:{}".format(len(X_test)))
    return X_train, y_train, X_test, y_test


