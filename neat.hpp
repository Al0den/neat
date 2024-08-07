#pragma once

#include <vector>
namespace neat {

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

    typedef struct Pair_s {
        Node *cur_node;
        int pos_x;
        int pos_y;
        int color_grad;

    } Pair;


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

            void randomizeNetwork();

            void mutateGenome();

            static double NEW_VALUE_PROBA;
            static double NEW_CONNECTION_PROBA;
            static double BIAS_FULL_PROBA;
            static double BIAS_CHANGE_PROBA;
            static double BIAS_FULL_RANGE;
            static double BIAS_SMALL_RANGE;
            static double WEIGHT_FULL_RANGE;
            static double WEIGHT_SMALL_RANGE;
            static double WEIGHT_FULL_PROBA;

            static double ADD_CONNECTION_RATIO;
            static double SPLIT_CONNECTION_RATIO;

            static double MUT_COUNT;
            static double MUT_PROBA;

            double *feed_forward(double *inputs);
            
            std::string serialize();
            void restore(std::string str);

            void mutateBias(double change_chance, double change_strength);
            void mutateConnectionWeight(double change_chance, double change_strength);
            void splitConnection(Connection *connection);
            void updateLayers();

            // Undeclared function, can be added by user 
            void display(int x, int y, int d_width, int d_height, void *renderer);
        private:
            int in_size;
            int out_size;

            bool no_layer_update;
            void newNode();
            void newConnection();
            void mutateBias();
            void mutateWeights();
    };

}
