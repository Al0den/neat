#include "./neat.hpp"
#include "./utils.hpp"

#include <cmath>
#include <assert.h>

using namespace neat;

double Network::SPLIT_CONNECTION_RATIO = 0.05;
double Network::ADD_CONNECTION_RATIO = 0.8;
double Network::MUT_PROBA = 0.25;
double Network::NEW_VALUE_PROBA = 0.2;
double Network::WEIGHT_FULL_RANGE = 1.0;
double Network::WEIGHT_SMALL_RANGE = 0.01;
double Network::WEIGHT_FULL_PROBA = 0.75;
double Network::BIAS_FULL_RANGE = 1.0;
double Network::BIAS_SMALL_RANGE = 0.01;
double Network::BIAS_FULL_PROBA = 0.25;
double Network::MUT_COUNT = 4;

double relu(double value) {
    return max(0, value);
}
double tanh_act(double value) {
    return tanh(value) * 2 / M_PI;
}
double none(double value) {
    return value;
}

Node::Node(double bias, int neuron_type) {
    this->bias = bias;
    this->neuron_type = neuron_type;

    this->current_value = 0;
    this->out_value = 0;

    if(neuron_type == INPUT_NEURON) {
        activationFunction = none;
        activationFunctionID = NONE;
    } else if(neuron_type == OUTPUT_NEURON) {
        activationFunction = tanh_act;
        activationFunctionID = TANH;
    } else {
        activationFunction = tanh_act;
        activationFunctionID = TANH;
    }

    output_id = -1;
    layer_id = -1;
}

Node::~Node() {}

int getNextId(std::vector<Connection *> connections) {
    if(connections.size() == 0) {
        return 0;
    }
    int highest = connections[0]->connection_id;
    for(Connection *connection : connections) {
        if(connection->connection_id > highest) {
            highest = connection->connection_id;
        }
    }
    return highest + 1;
}

Connection::Connection(double weight, int id, void *in, void *out) {
    this->weight = weight;
    this->connection_id = id;
    this->from = in;
    this->towards = out;
}

Connection::~Connection() {}

Network::Network(int in_size, int out_size, int hidden_layers) {
    this->in_size = in_size;
    this->out_size = out_size;

    no_layer_update = true; // Efficiency, we will add lots of neurons and connections
    for(int i=0; i<in_size; i++) {
        Node *node = new Node(0, INPUT_NEURON);
        this->nodes.push_back(node);
    }

    for(int i=0; i<hidden_layers; i++) {
        for(int j=0; j<in_size; j++) {
            Node *node = new Node(0, HIDDEN_NEURON);
            this->nodes.push_back(node);
        }
    }

    for(int i=0; i<out_size; i++) {
        Node *node = new Node(0, OUTPUT_NEURON);
        node->output_id = i;
        this->nodes.push_back(node);
        this->output_nodes.push_back(node);
    }

    for(int i=0; i<hidden_layers; i++) {
        for(int j=0; j<in_size; j++) {
            for(int k=0; k<in_size; k++) {
                addConnection(nodes[i * in_size + j], nodes[(i+1) * in_size + k], 0);
            }
        }
    }

    for(int i=0; i<in_size; i++) {
        for(int j=0; j<out_size; j++) {
            addConnection(nodes[hidden_layers * in_size + i], nodes[(hidden_layers + 1)* in_size + j], 0);
        }
    }
    no_layer_update = false;
    
    randomizeNetwork();
    updateLayers();

} 

Network::~Network() {
    for(Node *node : nodes) {
        delete node;
    }
    for (Connection *connection : connections) {
        delete connection;
    }
}

void Network::addConnection(Node *from, Node *towards, double weight) {
    Connection *connection = new Connection(weight, getNextId(connections), (void*)from, (void*)towards);
    for(int i=0; i<from->outgoing.size(); i++) {
        if(from->outgoing[i]->towards == towards) { // Duplicate connection, just change the weight
            from->outgoing[i]->weight = weight;
            return;
        }
    }
    if(from->neuron_type == towards->neuron_type && (from->neuron_type == INPUT_NEURON || from->neuron_type == OUTPUT_NEURON)) {
        return; // Cannot connect input to input or output to output
    }
    if(from->layer_id > towards->layer_id) {
        addConnection(towards, from, weight); // Add the opposite connection instead, as the from is deeper than towards
        return;
    }
    from->outgoing.push_back(connection);
    towards->incoming.push_back(connection);
    this->connections.push_back(connection);
    updateLayers();
}

void Network::randomizeNetwork() {
    RandomDoubleGenerator rdg(-1, 1);
    int n_n = nodes.size();
    for(int i=0; i<n_n; i++) {
        nodes[i]->bias = rdg();
    }

    int n_c = connections.size();
    for(int i=0; i<n_c; i++) {
        connections[i]->weight = rdg();
    }
    updateLayers();
}

void Network::removeConnection(Connection *connection) {
    for(int i=0; i<nodes.size(); i++) {
        Node *node = nodes[i];
        for(int j=0; j<node->outgoing.size(); j++) {
            if(node->outgoing[j] == connection) {
                node->outgoing.erase(node->outgoing.begin() + j);
                break;
            }
        }
        for(int j=0; j<node->incoming.size(); j++) {
            if(node->incoming[j] == connection) {
                node->incoming.erase(node->incoming.begin() + j);
                break;
            }
        }
    }
    for(int i=0; i<connections.size(); i++) {
        if(connections[i] == connection) {
            connections.erase(connections.begin() + i);
            break;
        }
    }

    updateLayers();

    delete connection;
}

void Network::removeNode(Node *node) {
    int n = nodes.size();
    
    n = node->outgoing.size();
    for(int i=0; i<n; i++) {
        removeConnection(node->outgoing[i]);
    }
    n = node->incoming.size();
    for(int i=0; i<n; i++) {
        removeConnection(node->incoming[i]);
    }   

    for(int i=0; i<n; i++) {
        if(nodes[i] == node) {
            nodes.erase(nodes.begin() + i);
            break;
        }
    }

    updateLayers();

    delete node;
}

double *Network::feed_forward(double *inputs) {
    int n_nodes = nodes.size();
    int n_layers = layers.size();

    for(int i=0; i<n_nodes; i++) {
        nodes[i]->current_value = 0;
        nodes[i]->out_value = 0;
    }

    for(int i=0; i<in_size; i++) {
        nodes[i]->current_value = inputs[i];
    }

    for(int i=0; i<n_layers; i++) {
        int n_local = layers[i].size();
        for(int j=0; j<n_local; j++) {
            Node *current = layers[i][j];
            current->out_value = current->activationFunction(current->bias + current->current_value);
            int n_connections = layers[i][j]->outgoing.size();
            for(int k=0; k<n_connections; k++) {
                Connection *connection = layers[i][j]->outgoing[k];
                ((Node *)(connection->towards))->current_value += current->out_value * connection->weight;
            }
        }
    }
    double *outputs = new double[out_size];

    for(int i=0; i<output_nodes.size(); i++) {
        outputs[output_nodes[i]->output_id] = output_nodes[i]->out_value;
    }
    return outputs;
}

void Network::splitConnection(Connection *connection) {
    Node *from = (Node *)connection->from;
    Node *towards = (Node *)connection->towards;
    Node *new_node = new Node(0, HIDDEN_NEURON);

    new_node->layer_id = from->layer_id + 1;
    towards->layer_id += 1;

    no_layer_update = true;
    addConnection(from, new_node, connection->weight);
    addConnection(new_node, towards, 1.0);
    removeConnection(connection);
    nodes.push_back(new_node);
    no_layer_update = false;

    updateLayers();
}

void Network::updateLayers() {
    if(no_layer_update) return;
    layers = std::vector<std::vector<Node*> >();
    std::vector<Node *> used;
    std::vector<Connection *> seen;
    std::vector<Node *> next;

    int current_layer_id = 0;

    int n = nodes.size();
    //Make sure input neurons are in the first layer
    for(int i=0; i<n; i++) {
        if(nodes[i]->neuron_type == INPUT_NEURON) {
            next.push_back(nodes[i]);
            nodes[i]->layer_id = current_layer_id;
        } else {
            nodes[i]->layer_id = -1;
        }
    }
    while(next.size() > 0) {
        layers.push_back(next);
        for(int i=0; i<next.size(); i++) {
            used.push_back(next[i]);
            int n = next[i]->outgoing.size();

            for(int j=0; j<n; j++) {
                Connection *connection = next[i]->outgoing[j];
                seen.push_back(connection);
            }
        }
        next = std::vector<Node*>();
        current_layer_id += 1;
        for(int i=0; i<nodes.size(); i++) {
            if(std::find(used.begin(), used.end(), nodes[i]) == used.end()) {
                bool no_incoming = true;
                int m = nodes[i]->incoming.size();
                for(int j=0; j<m; j++) {
                    if(std::find(seen.begin(), seen.end(), nodes[i]->incoming[j]) == seen.end()) {
                        no_incoming = false;
                    }
                }
                
                //Make sure output neurons are in the last layer
                if(no_incoming && nodes[i]->neuron_type != OUTPUT_NEURON) {
                    next.push_back(nodes[i]);
                    nodes[i]->layer_id = current_layer_id;
                }
            }
        }
    }
    next = std::vector<Node*>();
    current_layer_id += 1;
    for(int i=0; i<nodes.size(); i++) {
        if(nodes[i]->neuron_type == OUTPUT_NEURON) {
            next.push_back(nodes[i]);
            nodes[i]->layer_id = current_layer_id;
        } else if(nodes[i]->layer_id == -1) { // Node is not connected to the network
            nodes.erase(nodes.begin() + i);
        }
    }
    layers.push_back(next);
}

typedef struct Pair_s {
    Node *cur_node;
    int pos_x;
    int pos_y;
    int color_grad;

} Pair;

std::string Node::serialize() {
    return std::to_string(bias) + ";" + std::to_string(neuron_type) + ";" + std::to_string(output_id);
}

std::string Network::serialize() {
    updateLayers();
    std::string result = "";
    for(Node *node : nodes) {
        result += node->serialize() + ",";
    }
    result += "|";
    for (int j=0; j<connections.size(); j++) {
        Connection *connection = connections[j];
        // Find the index of the from and towards nodes
        int from = -1;
        int towards = -1;
        for(int i=0; i<nodes.size(); i++) {
            if(nodes[i] == (Node *)connection->from) {
                from = i;
            }
            if(nodes[i] == (Node *)connection->towards) {
                towards = i;
            }
        }
        if(from == -1 || towards == -1) {
            connections.erase(connections.begin() + j);
            continue;
        }
        result += std::to_string(connection->weight) + ";" + std::to_string(from) + ";" + std::to_string(towards) +  ",";
    }
    return result;
}

void Network::restore(std::string str) {
    nodes.clear();
    connections.clear();
    no_layer_update = true; // Efficiency, since we will add a lot of nodes and connections

    std::vector<std::string> parts;
    std::string current = "";
    for(int i=0; i<str.size(); i++) {
        if(str[i] == '|') {
            parts.push_back(current);
            current = "";
        } else {
            current += str[i];
        }
    }
    parts.push_back(current);
    assert (parts.size() == 2);

    std::vector<std::string> node_parts;
    current = "";
    for(int i=0; i<parts[0].size(); i++) {
        if(parts[0][i] == ',') {
            node_parts.push_back(current);
            current = "";
        } else {
            current += parts[0][i];
        }
    }

    for(int i=0; i<node_parts.size(); i++) {
        std::vector<std::string> node_data;
        current = "";
        for(int j=0; j<node_parts[i].size(); j++) {
            if(node_parts[i][j] == ';') {
                node_data.push_back(current);
                current = "";
            } else {
                current += node_parts[i][j];
            }
        }
        node_data.push_back(current);
        assert (node_data.size() == 3); 
        Node *node = new Node(std::stod(node_data[0]), std::stoi(node_data[1]));
        node->output_id = std::stod(node_data[2]);
        if(node->output_id != -1) {
            output_nodes.push_back(node);
        }
        nodes.push_back(node);
    }

    std::vector<std::string> connection_parts;
    current = "";
    for(int i=0; i<parts[1].size(); i++) {
        if(parts[1][i] == ',') {
            connection_parts.push_back(current);
            current = "";
        } else {
            current += parts[1][i];
        }
    }

    for(int i=0; i<connection_parts.size(); i++) {
        std::vector<std::string> connection_data;
        current = "";
        for(int j=0; j<connection_parts[i].size(); j++) {
            if(connection_parts[i][j] == ';') {
                connection_data.push_back(current);
                current = "";
            } else {
                current += connection_parts[i][j];
            }
        }
        connection_data.push_back(current);
        assert (connection_data.size() == 3);
        addConnection(nodes[std::stoi(connection_data[1])], nodes[std::stoi(connection_data[2])], std::stod(connection_data[0]));
    }

    no_layer_update = false;

    updateLayers();
}

void Network::mutateGenome() {
    RandomDoubleGenerator rdg(0, 1);
    for(int i=0; i<MUT_COUNT; i++) {
        if(rdg() < MUT_PROBA) {
            if(rdg() < 0.5) {
                mutateBias();
            } else {
                mutateWeights();
            }
        }
    }
    if(rdg() < SPLIT_CONNECTION_RATIO) {
        newNode();
    }
    if(rdg() < ADD_CONNECTION_RATIO) {
        newConnection();
    }
}

void Network::newNode() {
    if(connections.size() == 0) {
        return;
    }
    int rand_connection_id = randint(0, connections.size() - 1);
    Connection *connection = connections[rand_connection_id];
    splitConnection(connection);
}

void Network::newConnection() {
    if(nodes.size() < 2) {
        return;
    }
    int rand_node_id_1 = randint(0, nodes.size() - 1);
    int rand_node_id_2 = randint(0, nodes.size() - 1);
    Node *node1 = nodes[rand_node_id_1];
    Node *node2 = nodes[rand_node_id_2];
    RandomDoubleGenerator rdg(-1, 1);
    addConnection(node1, node2, rdg());
}

void Network::mutateBias() {
    int rand_node_id = randint(0, nodes.size() - 1);
    Node *node = nodes[rand_node_id];
    RandomDoubleGenerator rdg(0, 1);
    RandomDoubleGenerator rdg2(-1, 1);
    if(rdg() < NEW_VALUE_PROBA) {
        node->bias = rdg2();
    } else {
        if(rdg() < BIAS_FULL_PROBA) {
            node->bias += rdg2() * BIAS_FULL_RANGE;
        } else {
            node->bias += rdg2() * BIAS_SMALL_RANGE;
        }
    }
}

void Network::mutateWeights() {
    if(connections.size() == 0) {
        return;
    }
    int rand_connection_id = randint(0, connections.size() - 1);
    Connection *connection = connections[rand_connection_id];
    RandomDoubleGenerator rdg(0, 1);
    RandomDoubleGenerator rdg2(-1, 1);
    if(rdg() < NEW_VALUE_PROBA) {
        connection->weight = rdg2() * WEIGHT_FULL_RANGE;
    } else {
        if(rdg() < WEIGHT_FULL_PROBA) {
            connection->weight += rdg2() * WEIGHT_FULL_RANGE;
        } else {
            connection->weight += rdg2() * WEIGHT_SMALL_RANGE;
        }
    }
}

