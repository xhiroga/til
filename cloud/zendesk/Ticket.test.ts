import { Ticket } from "./ticket";


const ticket: Ticket = {
    url: 'https://example.zendesk.com/api/v2/tickets/1234.json',
    id: 1234,
    external_id: null,
    via: { channel: 'api', source: [] },
    created_at: '2020-11-05T11:05:38Z',
    updated_at: '2020-11-05T11:05:38Z',
    type: null,
    subject: 'りんごをたべるんご',
    raw_subject: 'りんごをたべるんご',
    description: 'たべるんご',
    priority: null,
    status: 'new',
    recipient: null,
    requester_id: 123456789012,
    submitter_id: 123456789012,
    assignee_id: null,
    organization_id: null,
    group_id: null,
    collaborator_ids: [],
    follower_ids: [],
    email_cc_ids: [],
    forum_topic_id: null,
    problem_id: null,
    has_incidents: false,
    is_public: true,
    due_at: null,
    tags: [],
    custom_fields: [],
    satisfaction_rating: null,
    sharing_agreement_ids: [],
    fields: [],
    followup_ids: [],
    brand_id: 360001196273,
    allow_channelback: false,
    allow_attachments: true
}

console.log(ticket.url)
