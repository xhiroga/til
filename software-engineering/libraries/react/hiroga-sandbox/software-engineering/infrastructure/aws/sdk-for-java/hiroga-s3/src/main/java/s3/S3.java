package s3;

import java.util.List;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.Bucket;
import software.amazon.awssdk.services.s3.model.ListBucketsRequest;
import software.amazon.awssdk.services.s3.model.ListBucketsResponse;

public class S3 {

    public static void main(String[] args) {
        Region region = Region.AP_NORTHEAST_1;
        S3Client s3Client = S3Client.builder().region(region).build();

        ListBucketsRequest req = ListBucketsRequest.builder().build();
        ListBucketsResponse res = s3Client.listBuckets(req);
        List<Bucket> buckets = res.buckets();
        for (Bucket bucket : buckets) {
            System.out.println(bucket.name());
        }
    }
}