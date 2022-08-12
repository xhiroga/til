package whoami

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/sts"
)

func Whoami() {
	// sessionの作成
	sess := session.Must(session.NewSessionWithOptions(session.Options{}))
	svc := sts.New(
		sess,
		aws.NewConfig().WithRegion("ap-northeast-1"),
	)
	input := &sts.GetCallerIdentityInput{}
	fmt.Println(svc.GetCallerIdentity(input))
}
