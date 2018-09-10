import argparse

from torch import nn
from torch import optim

import numpy as np

import utils
from model import Colorizer

IMG_SIZE      = 512
BATCH_SIZE    = 4
NUM_EPOCHS    = 4
LEARNING_RATE = 1e-2


def get_arguments():
    parser = argparse.ArgumentParser(description='train a neural network to colorize grayscale images')
    parser.add_argument('--source-dir', type=str, required=True,
                        help="path to directory containing source images")
    parser.add_argument('--img-size', type=int, default=IMG_SIZE,
                        help="side length for cropping and resizing source images; "
                             "default: {}".format(IMG_SIZE))
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help="batch size per iteration; default: {}".format(BATCH_SIZE))
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS,
                        help="number of training epochs; default: {}".format(NUM_EPOCHS))
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help="learning rate for RMSprop; default: {}".format(LEARNING_RATE))
    return parser.parse_args()


def train(source_dir, img_size, batch_size, num_epochs, learning_rate):
    # TODO: prepare test set
    inputs, targets = utils.prepare_training_data(source_dir, img_size)
    print("images ready")

    model = Colorizer(img_size)

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # TODO: learn how to use PyTorch's DataLoader
        # TODO: prepare validation set
        new_order = np.random.permutation(inputs.shape[0])
        inputs = inputs[new_order]
        targets = targets[new_order]

        # TODO: implement cross-validation

        for iteration, (batch_inputs, batch_targets) in enumerate(zip(inputs.split(batch_size),
                                                                      targets.split(batch_size))):
            optimizer.zero_grad()

            batch_outputs = model(batch_inputs).view(batch_targets.shape)
            loss = criterion(batch_outputs, batch_targets)

            loss.backward()
            optimizer.step()

            # TODO: make this nicer
            print(loss.item())
            utils.save(batch_outputs[0], "results/{}-{}.png".format(epoch, iteration))

    return model

if __name__ == "__main__":
    args = get_arguments()

    model = train(args.source_dir, args.img_size, args.batch_size, args.num_epochs, args.learning_rate)

    # TODO: implement testing data
