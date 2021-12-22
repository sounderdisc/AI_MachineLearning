import argparse
from TrainTest import runMNIST
from TrainTest import runCIFAR


runMINSTModel = True
runCIFARModel = True

if (runMINSTModel):
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=3,
                        help='Select mode between 1-3.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
                        
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    print("Starting MINST model")
    print("Mode: ", FLAGS.mode)
    print("LR: ", FLAGS.learning_rate)
    print("Batch size: ", FLAGS.batch_size)

    runMNIST(FLAGS)

# print a divider
if (runMINSTModel and runCIFARModel):
    print("")
    print("################################################################")
    print("")

if (runCIFARModel):
    # Now again for CIFAR
    # Set parameters for Sparse Autoencoder
    parser2 = argparse.ArgumentParser('CNN Exercise.')
    parser2.add_argument('--mode',
                        type=int, default=3,
                        help='Select mode between 1-3.')
    parser2.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser2.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to run trainer.')
    parser2.add_argument('--batch_size',
                        type=int, default=50,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser2.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
                        
    FLAGS = None
    FLAGS, unparsed = parser2.parse_known_args()

    print("Starting CIFAR model")
    print("Mode: ", FLAGS.mode)
    print("LR: ", FLAGS.learning_rate)
    print("Batch size: ", FLAGS.batch_size)

    runCIFAR(FLAGS)