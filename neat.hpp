#pragma once

#include <vector>
#include <SDL2/SDL.h>

enum {
    INPUT_NEURON, OUTPUT_NEURON, HIDDEN_NEURON
};

enum activation_functions {
    NONE, SIGMOID, TANH, RELU
};

class Connection {
    public:
        Connection(double weight, int id, void *in, void *out);
        ~Connection();

        double weight;
        int connection_id;
        void *towards;
        void *from;
};


class Node {
    public:
        Node(double bias, int neuron_type);
        ~Node();

        double bias;
        double current_value;
        double out_value;

        double (*activationFunction)(double);
        int activationFunctionID;

        std::vector<Connection *> outgoing;
        std::vector<Connection *> incoming;

        int neuron_type;
        int output_id;
        int layer_id;

        std::string serialize();
};

class Network {
    public:
        Network(int in_size, int out_size, int hidden_size);
        ~Network();

        std::vector<Node *> nodes;
        std::vector<Connection *> connections;
        std::vector<std::vector<Node*> > layers;
        std::vector<Node *> output_nodes;

        void addConnection(Node *in, Node *out, double weight);
        void removeConnection(Connection *connection);

        void removeNode(Node *node);

        void mutateBias(double change_chance, double change_strength);
        void mutateConnectionWeight(double change_chance, double change_strength);
        void splitConnection(Connection *connection);
        void updateLayers();

        void randomizeNetwork();
        void mutateGenome();

        int in_size;
        int out_size;

        bool no_layer_update;

        double *feed_forward(double *inputs);

        void display(int x, int y, int d_width, int d_height, SDL_Renderer *renderer);
        void custom_renderer(SDL_Renderer *renderer, SDL_Window *window);

        std::string serialize();
        void restore(std::string str);
    private:
        void newNode();
        void newConnection();
        void mutateBias();
        void mutateWeights();
};



