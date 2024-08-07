# NEAT Network implementation in C++

This is a simple yet efficient implementation of a NEAT neural network, in C++

## How to use

Simply copy the whole folder into root/third-party in your project, and include the neat.hpp file. Then, compile the neat.cpp file, and you should be good to go

The `neat.hpp` file exposes a Network class, which contains everything needed to use the Neural Network

For example:

```cpp
#include "neat.hpp"

using namespace neat;

int main() {
    //Initialize a network with 5 inputs, 2 outputs and 1 hidden layer
    Network *n = new Network(5, 2, 1);
    n->randomizeNetwork();
    
    //Mutate the network 10 times
    //Mutations probabilites are defined in neat.cpp, and can be changed
    for(int i=0; i<10; i++) {
        n->mutateGenome();
    }
    
    //Get NEAT output for a given input
    double inputs[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    double *output = n->feed_forward(&inputs[0]);

    //Note: You __must__ send an input with the correct size
    
    //serialize the network
    std::string serialized = n->serialize();

    delete n;

    //Restore the previously deleted network;
    Network *new_n = new Network(5, 2, 1);
    new_n->restore(serialized);

    delete new_n;
}
```
