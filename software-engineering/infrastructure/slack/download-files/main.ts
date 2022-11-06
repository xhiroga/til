import axios from 'axios';
import * as dotenv from 'dotenv';

const parsed = dotenv.config().parsed;
if (parsed === undefined) throw Error('Failed to parse .env file');
const { URL, TOKEN } = parsed;


// https://api.slack.com/types/file#authentication
axios.get(URL, {
  headers: {
    Authorization: `Bearer ${TOKEN}`
  },
  responseType: 'stream'
}).then(response => {
  streamToString(response.data).then(console.log)
})

function streamToString(stream) {
  const chunks = [];
  return new Promise((resolve, reject) => {
    stream.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
    stream.on('error', (err) => reject(err));
    stream.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
  })
}
