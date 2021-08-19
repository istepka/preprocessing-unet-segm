import train 
import matplotlib.pyplot as plt
import numpy as np







def predict(image):
    trainer = train.Trainer()
    trainer.build_model()

    trainer.model.load_weights('src/models/UNet_model_180821-Aug08.h5')

    result = trainer.model.predict(image)

    return result


def predict_showcase():
    trainer = train.Trainer()
    trainer.load_data()
    trainer.build_model()

    trainer.model.load_weights('src/models/UNet_model_190821:0000.h5')

    result = trainer.model.predict(trainer.validation_data[0])

    r = 1
    res = (result > 0.25).astype(float)
    print(type(result[r]), np.mean(res[r]))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Ground truth')
    ax.imshow(trainer.validation_data[r][1], cmap='gray')


    ax1 = fig.add_subplot(1,2,2)
    ax1.set_title('Predicted mask')
    ax1.imshow(res[r], cmap='gray')

    plt.show()


if __name__ == '__main__':
    predict_showcase()
