import axios from 'axios'
import { Ticket } from './ticket';
const update_ticket = () => {
  const axios = require("axios");

  const domain = `https://${process.env.ORGANIZATION}.zendesk.com`;

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

  const create_ticket_object = JSON.stringify({
    ticket: {
      subject: "りんごをたべるんご",
      comment: {
        body: "たべるんご",
      },
    },
  });

  const update_ticket_object = JSON.stringify({
    ticket: {
      subject: "やっぱりりんごをつくるんご", // this overwrite original ticket title
      comment: {
        body: "つくるんご", public: false
      },
    },
  });

  new Promise(async (resolve, reject) => {
    const create_url = `${domain}/api/v2/tickets.json`;
    const created_ticket: Ticket = await axios
      .post(create_url, create_ticket_object, config)
      .then((res) => {
        console.log("Creating tickets success!!!")
        console.log(res);
        const ticket: Ticket = res.data.ticket
        return ticket;
      })
      .catch((err) => {
        console.log(err);
        reject(err);
      });

    const update_url = `${domain}/api/v2/tickets/${created_ticket.id}.json`;
    const updated_ticket: Ticket = await axios
      .put(update_url, update_ticket_object, config)
      .then((res) => {
        console.log("Updating tickets success!!!")
        console.log(res);
        const ticket: Ticket = res.data.ticket
        return ticket;
      })
      .catch((err) => {
        console.log(err);
        reject(err);
      });

    resolve(updated_ticket);
  });
}

update_ticket()
// API Document
// https://developer.zendesk.com/rest_api/docs/support/tickets#update-ticket
