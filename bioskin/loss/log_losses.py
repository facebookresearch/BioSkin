# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def save_log_losses(learning_rate, writing_rate, epoch, file_log, writer,
                    losses_train, losses_param_train, losses_albedos_train, losses_albedos_full_train,
                    losses_test, losses_param_test, losses_albedos_test, losses_albedos_full_test
                    ):
    # log losses
    if epoch % writing_rate == 0:
        print((epoch, "--> learning rate", learning_rate))
        print((epoch, "train loss", losses_train[epoch]))
        print((epoch, "test loss ", losses_test[epoch]))
        print((epoch, "train loss param ", losses_param_train[epoch]))
        print((epoch, "test loss param ", losses_param_test[epoch]))
        print((epoch, "train loss albedo ", losses_albedos_train[epoch]))
        print((epoch, "test loss albedo ", losses_albedos_test[epoch]))
        print((epoch, "train loss albedo full ", losses_albedos_full_train[epoch]))
        print((epoch, "test loss albedo full ", losses_albedos_full_test[epoch]))

        file_log.write(str(epoch) + " --> learning rate " + str(learning_rate) + "\n")
        file_log.write(str(epoch) + " train loss " + str(losses_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss " + str(losses_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss param " + str(losses_param_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss param " + str(losses_param_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss albedo " + str(losses_albedos_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss albedo " + str(losses_albedos_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss albedo full " + str(losses_albedos_full_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss albedo full " + str(losses_albedos_full_test[epoch]) + "\n")

    writer.add_scalar('Loss/train', losses_train[epoch], epoch)
    writer.add_scalar('Loss/test', losses_test[epoch], epoch)
    writer.add_scalar('Loss/trainparam', losses_param_train[epoch], epoch)
    writer.add_scalar('Loss/testparam', losses_param_test[epoch], epoch)
    writer.add_scalar('Loss/trainalbedo', losses_albedos_train[epoch], epoch)
    writer.add_scalar('Loss/testalbedo', losses_albedos_test[epoch], epoch)
    writer.add_scalar('Loss/trainalbedofull', losses_albedos_full_train[epoch], epoch)
    writer.add_scalar('Loss/testalbedofull', losses_albedos_full_test[epoch], epoch)


def save_log_losses_exp(learning_rate, writing_rate, epoch, file_log, writer,
                    losses_train, losses_param_train, losses_exp_train, losses_albedos_train, losses_albedos_full_train,
                    losses_test, losses_param_test, losses_exp_test, losses_albedos_test, losses_albedos_full_test
                    ):
    # log losses
    if epoch % writing_rate == 0:
        print((epoch, "--> learning rate", learning_rate))
        print((epoch, "train loss", losses_train[epoch]))
        print((epoch, "test loss ", losses_test[epoch]))
        print((epoch, "train loss param ", losses_param_train[epoch]))
        print((epoch, "test loss param ", losses_param_test[epoch]))
        print((epoch, "train loss exp ", losses_exp_train[epoch]))
        print((epoch, "test loss exp ", losses_exp_test[epoch]))
        print((epoch, "train loss albedo ", losses_albedos_train[epoch]))
        print((epoch, "test loss albedo ", losses_albedos_test[epoch]))
        print((epoch, "train loss albedo full ", losses_albedos_full_train[epoch]))
        print((epoch, "test loss albedo full ", losses_albedos_full_test[epoch]))

        file_log.write(str(epoch) + " --> learning rate " + str(learning_rate) + "\n")
        file_log.write(str(epoch) + " train loss " + str(losses_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss " + str(losses_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss param " + str(losses_param_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss param " + str(losses_param_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss exp " + str(losses_exp_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss exp " + str(losses_exp_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss albedo " + str(losses_albedos_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss albedo " + str(losses_albedos_test[epoch]) + "\n")
        file_log.write(str(epoch) + " train loss albedo full " + str(losses_albedos_full_train[epoch]) + "\n")
        file_log.write(str(epoch) + " test loss albedo full " + str(losses_albedos_full_test[epoch]) + "\n")

    writer.add_scalar('Loss/train', losses_train[epoch], epoch)
    writer.add_scalar('Loss/test', losses_test[epoch], epoch)
    writer.add_scalar('Loss/trainparam', losses_param_train[epoch], epoch)
    writer.add_scalar('Loss/testparam', losses_param_test[epoch], epoch)
    writer.add_scalar('Loss/trainexp', losses_exp_train[epoch], epoch)
    writer.add_scalar('Loss/testexp', losses_exp_test[epoch], epoch)
    writer.add_scalar('Loss/trainalbedo', losses_albedos_train[epoch], epoch)
    writer.add_scalar('Loss/testalbedo', losses_albedos_test[epoch], epoch)
    writer.add_scalar('Loss/trainalbedofull', losses_albedos_full_train[epoch], epoch)
    writer.add_scalar('Loss/testalbedofull', losses_albedos_full_test[epoch], epoch)


def plot_losses(epoch, epoch_save, loss_graph_path, train_losses_global, test_losses_global):
    matplotlib.use("Agg")
    plt.plot(train_losses_global, 'g', label='Training loss')
    plt.plot(test_losses_global, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.ylim([0.0, 0.3])
    # if epoch == epoch_save:
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if epoch > 0 and epoch % epoch_save == 0:
        plt.savefig(loss_graph_path)

    plt.close('all')

    return 0


