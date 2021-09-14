from train_gpu import Trainer
import matplotlib.pyplot as plt
import numpy as np







def predict(image):
    trainer = Trainer()
    trainer.build_model()

    trainer.model.load_weights('src/models/UNet_model_256x256_02092021-181753.h5')

    result = trainer.model.predict(image)

    return result


def predict_showcase():
    trainer = Trainer()
    trainer.load_data()
    trainer.build_model()

    trainer.model.load_weights('src/models/UNet_model_256x256_01092021-192013.h5')

    result = trainer.model.predict(trainer.valid_iterator.__next__())

    r = 1
    res = (result > 0.25).astype(float)
    print(type(result[r]), np.mean(res[r]))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Ground truth')
    ax.imshow(trainer.test_mask_iterator.__next__(), cmap='gray')


    ax1 = fig.add_subplot(1,2,2)
    ax1.set_title('Predicted mask')
    ax1.imshow(res[r], cmap='gray')

    plt.show()


if __name__ == '__main__':
    #predict_showcase()

    images = np.load('npy_datasets/cv_train_images.npy')
    print(images.shape)
    # img, mask = images[0][0], images[1][0]

    # result = predict(img)

    # print(type(result))
