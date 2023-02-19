import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { v4 } from "uuid";
import { config } from 'dotenv'

config();
const run = async () => {
    const client = new S3Client({
        region: process.env.AWS_REGION,
    });
    const command = new PutObjectCommand({
        Bucket: process.env.S3_BUCKET,
        ContentType: "image/png",   // Not works.
        Key: v4(),
        Metadata: {
            "ContentType": "image/png", // become to `x-amz-meta-contenttype`
        }
    });
    const url = await getSignedUrl(client, command, { expiresIn: 3600 });
    console.log(url);
};
run();
