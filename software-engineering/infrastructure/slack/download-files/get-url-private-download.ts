import * as dotenv from 'dotenv';

const parsed = dotenv.config().parsed;
if (parsed === undefined) throw Error('Failed to parse .env file');
const { FILE_LINK, USER_TEAM } = parsed;

const [_https, _empty, _domain, _files, _user, file_id, file_name] = FILE_LINK.split('/')
const urlPrivate = `https://files.slack.com/files-pri/${USER_TEAM}-${file_id}/${file_name}`
const urlPrivateDownload = `https://files.slack.com/files-pri/${USER_TEAM}-${file_id}/download/${file_name}`

console.log('url_private:', urlPrivate)
console.log('url_private_download:', urlPrivateDownload)
