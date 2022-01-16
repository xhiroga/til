/**
 * Background Cloud Function to be triggered by Pub/Sub.
 *
 * @param {object} event The Cloud Functions event.
 */
exports.helloPubSub = async (event) => {
    const pubsubMessage = event.data;
    const name = pubsubMessage.data ? Buffer.from(pubsubMessage.data, 'base64').toString() : 'World';

    const greeting = `Hello, ${name}!`
    console.log(greeting);

    return greeting
};