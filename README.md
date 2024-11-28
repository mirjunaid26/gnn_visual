# Graph Neural Network (GNN) Visualization

This project provides an interactive web-based visualization of how a Graph Neural Network processes and learns from 4x4 matrices by treating them as graph structures. It demonstrates the power of GNNs in learning from both node features and graph topology.

## Live Demo

Visit [https://mirjunaid26.github.io/gnn_visual/](https://mirjunaid26.github.io/gnn_visual/) to see the visualization in action.

## Project Overview

The visualization demonstrates how a GNN processes 4x4 matrices by:
1. Converting each matrix into a graph structure
2. Applying message passing between nodes
3. Learning node embeddings
4. Using graph attention mechanisms
5. Making predictions based on the graph structure

## Features

### Data Representation
- Converts 4x4 matrices into graph structures
- Each matrix cell becomes a node with its value as a feature
- Edges connect adjacent cells
- Interactive graph visualization with Cytoscape.js

### GNN Architecture
- Input: 16 nodes (4x4 matrix cells)
- Graph Convolution Layers
- Attention Mechanism
- Dense output layer for binary classification

### Visualizations
1. **Matrix as Graph**
   - Interactive graph representation
   - Node colors indicate cell values
   - Toggleable edges and value labels
   - Grid layout matching matrix structure

2. **Node Embeddings**
   - 2D visualization of learned node features
   - Shows how node representations evolve
   - Color-coded by original values

3. **Message Passing**
   - Visualization of information flow
   - Highlighted active message paths
   - Real-time updates during training

4. **Graph Attention**
   - Shows attention weights between nodes
   - Indicates important connections
   - Updates during training

5. **Training Progress**
   - Loss plot
   - Accuracy plot
   - Training status updates

## Technology Stack

- **TensorFlow.js**: Neural network implementation
- **Cytoscape.js**: Graph visualization
- **Plotly.js**: Data plotting
- **HTML/CSS/JavaScript**: Web interface

## Implementation Details

### JavaScript (web_gnn.js)
- `matrixToGraph()`: Converts matrices to graph structures
- `createModel()`: Defines the GNN architecture
- `updateGraphVisualization()`: Handles graph updates
- `trainModel()`: Manages training and visualization updates

### HTML (index.html)
- Responsive layout for visualizations
- Interactive controls for graph display
- Status updates and progress tracking

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/mirjunaid26/gnn_visual.git
   ```

2. Navigate to the project directory:
   ```bash
   cd gnn_visual
   ```

3. Serve the files using any HTTP server. For example, using Python:
   ```bash
   python -m http.server
   ```

4. Open your browser and visit `http://localhost:8000`

## Browser Requirements

- Modern web browser with JavaScript enabled
- WebGL support for TensorFlow.js
- Sufficient RAM for training (recommended: 4GB+)

## Understanding the Visualization

1. **Graph Structure**
   - Each node represents a cell from the 4x4 matrix
   - Edges connect adjacent cells
   - Node colors indicate cell values
   - Toggle edges and values for different views

2. **Training Process**
   - Watch how node embeddings evolve
   - See message passing in action
   - Observe attention weights change
   - Monitor training metrics

3. **Interpreting Results**
   - Node positions in embedding space show learned features
   - Edge colors in attention view show important connections
   - Training plots show learning progress

## Contributing

Feel free to open issues or submit pull requests for improvements. Some areas for potential enhancement:

- Additional graph neural network architectures
- More complex graph structures
- Interactive hyperparameter adjustment
- Advanced visualization features
- Performance optimizations

## License

This project is open source and available under the MIT License.

## Acknowledgments

- TensorFlow.js team for the deep learning framework
- Cytoscape.js team for the graph visualization library
- Plotly team for the plotting tools
- The GNN research community for inspiration and knowledge sharing
