import { Octokit } from "@octokit/core";
import { restEndpointMethods } from "@octokit/plugin-rest-endpoint-methods";
import { getJukoushaStartupPhase, getSessionAbstract, getSessionFormat, getSessionHosoku, getSessionTopic, getToudanshaStartupPhase, getTwitterId } from "./util";

const MyOctotKit = Octokit.plugin(restEndpointMethods);
const octokit = new MyOctotKit()

const response = await octokit.rest.issues.listForRepo({ owner: "aws-startup-community", repo: "aws-startup-community-conference-2022-cfp", state: "open" })

const csv = response.data.map(({ title, body }) => {
    if (!body) {
        return
    }
    body = body.replace(/\r\n/g, "\n")
    const twitterId = getTwitterId(body)
    const sessionAbstract = getSessionAbstract(body)
    const sessionHosoku = getSessionHosoku(body)
    const toudanshaStartupPhase = getToudanshaStartupPhase(body)
    const jukoushaStartupPhase = getJukoushaStartupPhase(body)
    const sessionTopic = getSessionTopic(body)
    const sessionFormat = getSessionFormat(body)
    return `"${title}","${twitterId}","${sessionAbstract}","${sessionHosoku}","${toudanshaStartupPhase}","${jukoushaStartupPhase}","${sessionTopic}","${sessionFormat}"`
}).join("\n")
console.log("タイトル,Twitter ID,アブストラクト,補足,所属Startupフェーズ,受講者スタートアップフェーズ,トピック,フォーマット")
console.log(csv)
