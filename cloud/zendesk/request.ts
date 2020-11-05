import axios from 'axios'
const request_ticket = () => {
  const request_url = `https://${process.env.ORGANIZATION}.zendesk.com/api/v2/requests.json`;

  const basic_token = Buffer.from(
    `${process.env.ZENDESK_EMAIL}/token:${process.env.ZENDESK_TOKEN}`
  ).toString("base64");
  console.log(`basic_token: ${basic_token}`);
  const config = {
    headers: {
      Authorization: `Basic ${basic_token}`,
      "Content-Type": "application/json",
    },
  };
  const ticket_object = JSON.stringify({
    request: {
      subject: "みかんに関するお問い合わせ",
      requester: { name: "みかん侍", email: "mikan@example.com" },
      comment: {
        body:
          "改行のテストです\r\n↑ここにrn改行があるはず\n↑ここにn改行\rここにr改行",
      },
    },
  });

  axios
    .post(request_url, ticket_object, config)
    .then((res) => {
      console.log(res.data);
    })
    .catch((err) => {
      console.log(err);
    });
}

request_ticket()

// API Document
// https://developer.zendesk.com/rest_api/docs/support/requests#create-request
