package cc.hiroga

data class SlackResponse(
    val response_type: String,
    val text: String,
    val attachments: Array<SlackResponseAttachement>
)
