// Initialize arrays for storing training history
let trainLosses = [];
let trainAccs = [];
let valLosses = [];
let valAccs = [];

// Global variables for visualization state
let showEdges = true;
let showValues = true;
let cy; // Cytoscape instance

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

// Convert matrix to graph structure
function matrixToGraph(matrix) {
    const nodes = [];
    const edges = [];
    
    // Create nodes
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const nodeId = `n${i}-${j}`;
            nodes.push({
                data: {
                    id: nodeId,
                    value: matrix[i][j],
                    label: `(${i},${j}): ${matrix[i][j].toFixed(2)}`
                }
            });
            
            // Create edges to adjacent nodes
            if (i > 0) edges.push({ data: { source: nodeId, target: `n${i-1}-${j}` } });
            if (j > 0) edges.push({ data: { source: nodeId, target: `n${i}-${j-1}` } });
        }
    }
    
    return { nodes, edges };
}

// Initialize Cytoscape graph
function initializeGraph(matrix) {
    const graphData = matrixToGraph(matrix);
    
    cy = cytoscape({
        container: document.getElementById('graph-container'),
        elements: {
            nodes: graphData.nodes,
            edges: graphData.edges
        },
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'data(value)',
                    'label': showValues ? 'data(label)' : '',
                    'color': '#fff',
                    'text-outline-color': '#000',
                    'text-outline-width': 2,
                    'font-size': '12px'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#666',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'grid',
            rows: 4,
            cols: 4
        }
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
        const i = parseInt(id.split('-')[0].substring(1));
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
        const [matrices, labels] = generateData(100);
        
        // Initialize graph with first matrix
        initializeGraph(matrices[0]);
        
        status.textContent = 'Creating model...';
        const model = createModel();
        
        // Display model summary
        model.summary();
        
        status.textContent = 'Training model...';
        
        // Convert matrices to tensors
        const xTrain = tf.tensor2d(matrices.map(m => m.flat()));
        const yTrain = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);
        
        // Train the model
        await model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    trainLosses.push(logs.loss);
                    trainAccs.push(logs.acc);
                    
                    status.textContent = `Training... Epoch ${epoch + 1}/50`;
                    plotTrainingProgress();
                    
                    // Update visualizations every few epochs
                    if (epoch % 5 === 0) {
                        // Get embeddings for current matrix
                        const currentMatrix = matrices[0];
                        const embeddings = await tf.tidy(() => {
                            const input = tf.tensor2d([currentMatrix.flat()]);
                            return model.layers[0].apply(input).arraySync()[0];
                        });
                        
                        // Update visualizations
                        updateGraphVisualization(currentMatrix, embeddings);
                        plotEmbeddings(embeddings.map((e, i) => [e, embeddings[(i + 1) % embeddings.length]]));
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
