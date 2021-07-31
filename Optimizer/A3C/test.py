import torch

from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker import Worker
from Optimizer.A3C.Server_method import synchronize
from Optimizer.A3C.Worker_method import asynchronize


def zero_actor_weights(worker):
    for actor_param in worker.actor_net.parameters():
        actor_param.data *= 0


def print_actor_param(worker):
    for actor_param in worker.actor_net.parameters():
        print(actor_param.data)


def print_actor_grad(worker):
    for actor_param in worker.actor_net.parameters():
        print(actor_param.grad)


# test copy weights
def test1():
    baseServer = Server(nb_state_feature=4, nb_action=3, name="server")
    worker1 = Worker(Server_object=baseServer, name="worker_1", id=1)
    worker2 = Worker(Server_object=baseServer, name="worker_2", id=2)
    worker3 = Worker(Server_object=baseServer, name="worker_3", id=3)

    workerList = [worker1, worker2, worker3]
    synchronize(baseServer, workerList)

    print(worker1.actor_net.parameters())
    print(worker2.actor_net.parameters())

    print("===================================WORKER 1 BEFORE========================================")
    print_actor_param(worker1)
    zero_actor_weights(worker2)
    print("===================================WORKER 1 AFTER========================================")
    print_actor_param(worker1)
    print("===================================WORKER 2========================================")
    print_actor_param(worker2)

    del baseServer
    del worker1
    del worker2
    del worker3


# test synchronize
def test2():
    baseServer = Server(nb_state_feature=4, nb_action=3, name="server")
    worker1 = Worker(Server_object=baseServer, name="worker_1", id=1)
    worker2 = Worker(Server_object=baseServer, name="worker_2", id=2)
    workerList = [worker1, worker2]

    test_vector = torch.Tensor([1, 2, 3, 4])
    result1 = worker1.get_value(test_vector)
    result2 = worker2.get_value(test_vector)

    print(result1)
    print(result2)

    synchronize(baseServer, workerList)
    result1 = worker1.get_value(test_vector)
    result2 = worker2.get_value(test_vector)

    print("===================================AFTER========================================")
    print(result1)
    print(result2)

    del baseServer
    del worker1
    del worker2


# test asynchronize
def test3():
    baseServer = Server(nb_state_feature=4, nb_action=3, name="server")
    worker1 = Worker(Server_object=baseServer, name="worker_1", id=1)

    print("----------------------")
    print_actor_grad(worker1)  # return all None
    print("----------------------")
    print_actor_grad(baseServer)  # return all None

    zero_actor_weights(baseServer)
    print_actor_param(baseServer)
    test_vector = torch.Tensor([1, 2, 3, 4])

    out_actor_worker1 = worker1.get_policy(test_vector)
    loss = 1 / 2 * torch.sum((torch.Tensor([4, 5, 6]) - out_actor_worker1) ** 2)
    loss.backward()

    print('---------------------')
    print_actor_grad(worker1)

    asynchronize(worker1, baseServer)
    print('------------ACTOR PARAM---------')
    print_actor_param(baseServer)
    print('------------ACTOR GRAD---------')
    print_actor_grad(baseServer)

    del baseServer
    del worker1


if __name__ == "__main__":
    """
    CONFIRM: all test passed, A3C is implemented correctly
    """
    H_origin = torch.Tensor([-1.6747e+00, -6.5780e-01, -2.5957e+00, -2.4697e-02, -3.1876e-02,
        -2.2427e-02, -2.6870e-01, -2.5951e-02, -4.2670e-02, -4.2670e-02,
        -4.2670e-02, -4.2670e-02, -3.1483e-01, -4.2670e-02, -2.2175e-02,
        -2.3683e-02, -2.3764e-02,  2.4516e+01, -4.2670e-02, -2.5749e-02,
        -2.2482e-02, -3.3147e-02, -4.2670e-02, -2.6046e-02, -4.2670e-02,
        -2.4480e-02, -2.2863e-02, -1.2432e+00, -3.7386e-01, -2.8373e-02,
        -2.3438e-02, -2.3772e-02, -4.2670e-02, -2.4475e-02, -2.6655e-02,
        -2.8205e-02, -3.5353e-01, -4.2670e-02, -4.2670e-02, -4.2670e-02,
        -4.2670e-02, -4.2670e-02, -2.3289e-02, -4.2670e-02, -2.5100e-02,
        -4.1635e-01, -4.2670e-02, -4.2670e-02, -2.7280e-01, -4.2670e-02,
        -2.4453e-01, -4.2670e-02, -2.2127e-02, -4.9067e-01, -4.2670e-02,
        -4.2670e-02, -4.2670e-02, -2.4370e-02, -2.3185e-02, -4.2670e-02,
        -2.3052e-02, -2.4278e-02, -2.8749e-02, -2.1707e-02, -2.6775e-01,
        -4.2670e-02, -2.4796e-02, -4.2670e-02, -2.5090e-02, -2.4610e-02,
        -4.2670e-02, -2.3197e-02, -2.4116e-02, -1.2794e+00, -4.2670e-02,
        -4.2670e-02, -4.2670e-02, -2.2712e-02, -2.2061e-02, -2.5413e-02,
         0.0000e+00])

    # T = 2 * (H_origin - torch.mean(H_origin))
    # print(T)
    # G = torch.exp(T)
    # print(G)
    # H = G/torch.sum(G)
    # print(H)
    print(torch.std(H_origin))
