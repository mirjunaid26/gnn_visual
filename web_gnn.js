// Initialize arrays for storing training history
let trainLosses = [];
let trainAccs = [];
let valLosses = [];
let valAccs = [];
let attentionWeights = [];

// Global variables for visualization state
let showEdges = true;
let showValues = true;
let cy; // Cytoscape instance

// Style for the graph visualization
const graphStyle = [
    {
        selector: 'node',
        style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'width': 50,
            'height': 50,
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '12px',
            'color': '#fff',
            'text-outline-width': 1,
            'text-outline-color': '#000'
        }
    },
    {
        selector: 'edge',
        style: {
            'width': 'data(weight)',
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'curve-style': 'bezier',
            'opacity': 0.7
        }
    },
    {
        selector: '.highlighted',
        style: {
            'background-color': '#ff0',
            'line-color': '#ff0',
            'transition-property': 'background-color, line-color',
            'transition-duration': '0.5s'
        }
    },
    {
        selector: '.message',
        style: {
            'width': 20,
            'height': 20,
            'background-color': '#ff0',
            'opacity': 0.8
        }
    }
];

// Helper function to generate random 4x4 matrices
function generateData(numSamples) {
    try {
        const data = [];
        const labels = [];
        
        for (let i = 0; i < numSamples; i++) {
            // Create a 4x4 matrix with random values between 0 and 1
            const matrix = Array.from({ length: 4 }, () => 
                Array.from({ length: 4 }, () => Math.random())
            );
            
            // Calculate sum and create label (1 if sum > 8, 0 otherwise)
            const sum = matrix.flat().reduce((a, b) => a + b, 0);
            const label = sum > 8 ? 1 : 0;
            
            data.push(matrix);
            labels.push(label);
        }
        
        return [data, labels];
    } catch (error) {
        console.error('Error in generateData:', error);
        throw error;
    }
}

// Convert matrix to graph data for Cytoscape
function matrixToGraph(matrix) {
    const nodes = [];
    const edges = [];
    
    // Create nodes
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const value = matrix[i][j];
            const color = valueToColor(value);
            nodes.push({
                data: {
                    id: `${i}-${j}`,
                    label: value.toFixed(2),
                    color: color,
                    value: value,
                    row: i,
                    col: j
                },
                position: {
                    x: j * 100 + 100,
                    y: i * 100 + 100
                }
            });
        }
    }
    
    // Create edges between adjacent cells
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const directions = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]];
            for (const [di, dj] of directions) {
                const ni = i + di;
                const nj = j + dj;
                if (ni >= 0 && ni < 4 && nj >= 0 && nj < 4) {
                    edges.push({
                        data: {
                            id: `${i}-${j}_${ni}-${nj}`,
                            source: `${i}-${j}`,
                            target: `${ni}-${nj}`,
                            weight: 2,
                            color: '#ccc'
                        }
                    });
                }
            }
        }
    }
    
    return { nodes, edges };
}

// Convert value to color using a color scale
function valueToColor(value) {
    const scale = d3.scaleSequential(d3.interpolateViridis)
        .domain([0, 1]);
    return scale(value);
}

// Initialize Cytoscape graph
function initGraph(matrix) {
    const container = document.getElementById('graph-container');
    const graphData = matrixToGraph(matrix);
    
    cy = cytoscape({
        container: container,
        elements: {
            nodes: graphData.nodes,
            edges: graphData.edges
        },
        style: graphStyle,
        layout: {
            name: 'preset'
        },
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false
    });
    
    return cy;
}

// Animate message passing between nodes
function animateMessagePassing() {
    const nodes = cy.nodes().toArray();
    let currentIndex = 0;
    
    function highlightNextNode() {
        if (currentIndex > 0) {
            nodes[currentIndex - 1].removeClass('highlighted');
            nodes[currentIndex - 1].connectedEdges().removeClass('highlighted');
        }
        
        if (currentIndex < nodes.length) {
            const currentNode = nodes[currentIndex];
            currentNode.addClass('highlighted');
            currentNode.connectedEdges().addClass('highlighted');
            
            currentIndex++;
            setTimeout(highlightNextNode, 500);
        } else {
            // Reset for next animation
            nodes[nodes.length - 1].removeClass('highlighted');
            nodes[nodes.length - 1].connectedEdges().removeClass('highlighted');
            currentIndex = 0;
        }
    }
    
    highlightNextNode();
}

// Update edge weights based on attention weights
function updateAttentionWeights(weights) {
    const maxWeight = Math.max(...weights);
    const scale = d3.scaleSequential(d3.interpolateReds)
        .domain([0, maxWeight]);
    
    cy.edges().forEach((edge, i) => {
        const weight = weights[i];
        edge.data('weight', weight * 5);
        edge.data('color', scale(weight));
    });
}

// Create GNN model
function createModel() {
    try {
        const model = tf.sequential();
        
        // Graph Convolution Layer 1
        model.add(tf.layers.dense({
            inputShape: [16], // Flattened 4x4 matrix
            units: 32,
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        
        // Graph Convolution Layer 2
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu'
        }));
        
        // Output Layer
        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax'
        }));
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    } catch (error) {
        console.error('Error in createModel:', error);
        throw error;
    }
}

// Update graph visualization
function updateGraphVisualization(matrix, embeddings) {
    try {
        // Update node colors based on current values
        cy.nodes().forEach((node, index) => {
            const i = Math.floor(index / 4);
            const j = index % 4;
            const value = matrix[i][j];
            const embedding = embeddings[index];
            
            node.style({
                'background-color': `rgb(${Math.floor(value * 255)}, ${Math.floor(value * 255)}, ${Math.floor(value * 255)})`,
                'label': showValues ? `(${i},${j}): ${value.toFixed(2)}` : ''
            });
        });
        
        // Update edge visibility
        cy.edges().style('display', showEdges ? 'element' : 'none');
    } catch (error) {
        console.error('Error updating graph:', error);
    }
}

// Plot embeddings
function plotEmbeddings(embeddings) {
    try {
        const trace = {
            x: embeddings.map(e => e[0]),
            y: embeddings.map(e => e[1]),
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: embeddings.map(e => e[0] * 255),
                colorscale: 'Viridis'
            }
        };
        
        const layout = {
            title: 'Node Embeddings',
            xaxis: { title: 'Dimension 1' },
            yaxis: { title: 'Dimension 2' }
        };
        
        Plotly.newPlot('embeddings-plot', [trace], layout);
    } catch (error) {
        console.error('Error plotting embeddings:', error);
    }
}

// Plot training progress
function plotTrainingProgress() {
    try {
        // Plot loss
        const lossTrace = {
            y: trainLosses,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss'
        };
        
        const lossLayout = {
            title: 'Training Loss',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' }
        };
        
        Plotly.newPlot('loss-plot', [lossTrace], lossLayout);
        
        // Plot accuracy
        const accTrace = {
            y: trainAccs,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Accuracy'
        };
        
        const accLayout = {
            title: 'Training Accuracy',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Accuracy' }
        };
        
        Plotly.newPlot('accuracy-plot', [accTrace], accLayout);
    } catch (error) {
        console.error('Error plotting training progress:', error);
    }
}

// Toggle edge visibility
function toggleEdges() {
    showEdges = !showEdges;
    cy.edges().style('display', showEdges ? 'element' : 'none');
}

// Toggle value labels
function toggleValues() {
    showValues = !showValues;
    cy.nodes().style('label', node => {
        if (!showValues) return '';
        const id = node.id();
        const i = parseInt(id.split('-')[0]);
        const j = parseInt(id.split('-')[1]);
        return `(${i},${j}): ${node.data('value').toFixed(2)}`;
    });
}

// Training function
async function trainModel() {
    try {
        const status = document.getElementById('status');
        status.textContent = 'Generating training data...';
        
        // Generate training data
        const [matrices, labels] = generateData(50); // Reduced from 100 to 50
        
        // Initialize graph with first matrix
        initGraph(matrices[0]);
        
        status.textContent = 'Creating model...';
        const model = createModel();
        
        // Display model summary
        model.summary();
        
        status.textContent = 'Training model...';
        
        // Convert matrices to tensors
        const xTrain = tf.tensor2d(matrices.map(m => m.flat()));
        const yTrain = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);
        
        // Train the model with reduced epochs and update frequency
        await model.fit(xTrain, yTrain, {
            epochs: 20, // Reduced from 50 to 20
            batchSize: 16, // Reduced from 32 to 16
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    trainLosses.push(logs.loss);
                    trainAccs.push(logs.acc);
                    
                    status.textContent = `Training... Epoch ${epoch + 1}/20`;
                    
                    // Update visualizations less frequently
                    if (epoch % 10 === 0) { // Changed from 5 to 10
                        plotTrainingProgress();
                        
                        // Get embeddings for current matrix
                        const currentMatrix = matrices[0];
                        const embeddings = await tf.tidy(() => {
                            const input = tf.tensor2d([currentMatrix.flat()]);
                            return model.layers[0].apply(input).arraySync()[0];
                        });
                        
                        // Update visualizations
                        updateGraphVisualization(currentMatrix, embeddings);
                        plotEmbeddings(embeddings.map((e, i) => [e, embeddings[(i + 1) % embeddings.length]]));
                        
                        // Add delay between updates to avoid rate limiting
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                    
                    await tf.nextFrame();
                }
            }
        });
        
        status.textContent = 'Training complete!';
        
    } catch (error) {
        console.error('Error during training:', error);
        status.textContent = 'Error during training: ' + error.message;
    }
}

// Start training when page loads
window.addEventListener('load', async () => {
    console.log('Page loaded, checking dependencies...');
    try {
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js not loaded');
        }
        console.log('TensorFlow.js version:', tf.version.tfjs);
        
        if (typeof Plotly === 'undefined') {
            throw new Error('Plotly not loaded');
        }
        console.log('Plotly loaded successfully');
        
        if (typeof cytoscape === 'undefined') {
            throw new Error('Cytoscape.js not loaded');
        }
        console.log('Cytoscape.js loaded successfully');
        
        await trainModel();
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('status').textContent = 'Error: ' + error.message;
    }
});
