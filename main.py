import torch

from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets

from utils import load_network, get_device, parse_args, UnknownNetworkException


def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    # Load the network
    network = load_network(network_name=args.network_name,
                           device=device)
    network.eval()

    # Load the dataset
    if 'ResNet' in args.network_name:
        _, _, loader = load_CIFAR10_datasets(test_batch_size=args.batch_size)
    else:
        loader = load_ImageNet_validation_set(batch_size=args.batch_size,
                                              image_per_class=1)

    # Generate fault list
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       device=device)

    fault_list = fault_manager.get_weight_fault_list(load_fault_list=True,
                                                     save_fault_list=True)


    # Execute the fault injection campaign with the smart network
    fault_injection_executor = FaultInjectionManager(network=network,
                                                     network_name=args.network_name,
                                                     device=device,
                                                     loader=loader)

    fault_injection_executor.run_clean()
    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list)


if __name__ == '__main__':
    main(args=parse_args())
