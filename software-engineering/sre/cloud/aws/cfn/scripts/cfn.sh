cmd=$1
stack_name=$2

if [ -z $cmd ]; then
    echo "Error: Command is required"
    exit -1
elif [ $cmd = "create" ]; then
    case $3 in
        "") echo "Error: Please give a template file path!"; exit 0;;
        /*) absolute=$3 ;;
        *) absolute=$PWD/$3 ;;
        # S3にはこれから対応する、かも
    esac
    aws cloudformation create-stack --stack-name $stack_name --template-body file://$absolute
elif [ $cmd = "delete" ]; then
    aws cloudformation delete-stack --stack-name $stack_name
elif [ $cmd = "status" ]; then
    aws cloudformation describe-stacks --stack-name $stack_name | jq .Stacks[] | jq '.StackId, .StackStatus'
fi