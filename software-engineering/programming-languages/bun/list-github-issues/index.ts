import { Octokit } from "@octokit/core";

const octokit = new Octokit();

const response = await octokit.request("GET /repos/{org}/{repo}/issues", {
  org: "aws-startup-community",
  repo: "aws-startup-community-conference-2022-cfp",
  type: "private",
});

console.log(response)
