import * as tf from '@tensorflow/tfjs';
import fetch from 'node-fetch';

export default async function handler(req, res) {
    try {
        // Cargar el modelo desde una URL externa
        const modelUrl = 'model.json';
        const model = await tf.loadGraphModel(modelUrl);

        // Cargar el scaler y el label encoder desde URLs
        const scalerUrl = 'scaler.json';
        const labelEncoderUrl = 'label_encoder.json';

        const [scalerResponse, labelEncoderResponse] = await Promise.all([
            fetch(scalerUrl),
            fetch(labelEncoderUrl)
        ]);

        const scaler = await scalerResponse.json();
        const labelEncoder = await labelEncoderResponse.json();

        // Obtener parámetros de consulta
        const { hum, luz, pres, temp, vel } = req.query;

        // Validación de parámetros
        if ([hum, luz, pres, temp, vel].some(val => isNaN(parseFloat(val)))) {
            return res.status(400).json({ error: 'Invalid input parameters. Ensure all parameters are numbers.' });
        }

        // Preparar la entrada para la predicción
        const input = [parseFloat(hum), parseFloat(luz), parseFloat(pres), parseFloat(temp), parseFloat(vel)];
        const scaledInput = input.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);

        // Realizar la predicción
        const tensorInput = tf.tensor2d([scaledInput], [1, 5]);
        const prediction = model.predict(tensorInput);
        const predClass = prediction.argMax(-1).dataSync()[0];
        const result = labelEncoder.classes[predClass];

        // Devolver la predicción
        return res.status(200).json({ prediction: result });

    } catch (error) {
        console.error('Error during prediction:', error);
        return res.status(500).json({ error: `Error during prediction: ${error.message}` });
    }
}
