import { createPresignedPost } from "@aws-sdk/s3-presigned-post";
import { S3Client } from "@aws-sdk/client-s3";
import { v4 } from "uuid";
import { config } from 'dotenv'

config();
const run = async () => {
    const client = new S3Client({ region: process.env.AWS_REGION });
    const Bucket = process.env.S3_BUCKET!!;
    const Key = v4();
    const Fields = {
        acl: "public-read",
        "Content-Type": "image/png",
    };
    const { url, fields } = await createPresignedPost(client, {
        Bucket,
        Key,
        Conditions: [],
        Fields,
        Expires: 600, //Seconds before the presigned post expires. 3600 by default.
    });
    console.log(url, fields);
}
run();
