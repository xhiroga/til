# Cognito UserPool Federation機能の仕様調査

# Cognito UserPoolの作成
## ユーザーの連絡先を検証しない。
![](https://gyazo.com/21c0104f592ac5a28895309cfc41abaa.png)

### メールアドレスを属性に持たないユーザーがサインアップした場合、そのユーザーはConfirmを無視してサインアップ可能？  
不可能。管理者によるユーザーの検証が必要。  

# Cognito UserPoolの使用
### Amplify.configure()の設定
https://aws-amplify.github.io/docs/js/authentication#configure-your-app

## Auth.signIn()
### Auth.signIn()の返り値
```
CognitoUser {
    username: '******',
    pool:
     CognitoUserPool {
       userPoolId: 'ap-northeast-1_******',
       clientId: '5iletqtg1b0emikjplh******',
       client:
        Client {
          endpoint: 'https: //cognito-idp.ap-northeast-1.amazonaws.com/',
          userAgent: 'aws-amplify/0.1.x js'
        },
       advancedSecurityDataCollectionFlag: true,
       storage: {
            [Function: MemoryStorage
            ]
          setItem: [Function
            ],
          getItem: [Function
            ],
          removeItem: [Function
            ],
          clear: [Function
            ]
        }
    },
    Session: null,
    client:
     Client {
       endpoint: 'https: //cognito-idp.ap-northeast-1.amazonaws.com/',
       userAgent: 'aws-amplify/0.1.x js'
    },
    signInUserSession:
     CognitoUserSession {
       idToken:
        CognitoIdToken {
          jwtToken:
           '******',
          payload: [Object
            ]
        },
       refreshToken:
        CognitoRefreshToken {
          token:
           '******'
        },
       accessToken:
        CognitoAccessToken {
          jwtToken:
           '******',
          payload: [Object
            ]
        },
       clockDrift: 0
    },
    authenticationFlowType: 'USER_SRP_AUTH',
    storage: {
        [Function: MemoryStorage
        ]
       setItem: [Function
        ],
       getItem: [Function
        ],
       removeItem: [Function
        ],
       clear: [Function
        ]
    },
    keyPrefix: 'CognitoIdentityServiceProvider.5iletqtg1b0emikjplh5******',
    userDataKey:
     'CognitoIdentityServiceProvider.5iletqtg1b0emikjplh5******.280963.userData',
    deviceKey: undefined,
    attributes: { sub: '0859c043-3813-4280-ad1a-731517578259'
    },
    preferredMFA: 'NOMFA'
}
```

### Confirmされていないユーザーがログインしようとした場合は？
```
{ code: 'UserNotConfirmedException',
  name: 'UserNotConfirmedException',
  message: 'User is not confirmed.' }
```



## Auth.signUp()
### サインアップ直後の状態は？  
![](https://gyazo.com/382d0617c983935556028a1c2fa2ef89.png)

### Auth.signUp()の返り値は？  
```
{
    user:
    CognitoUser {
    username: '******',
    pool: [CognitoUserPool],
    Session: null,
    client: [Client],
    signInUserSession: null,
    authenticationFlowType: 'USER_SRP_AUTH',
    storage: [Function],
    keyPrefix: 'CognitoIdentityServiceProvider.e33qfsolr15a3mvdb12******',
    userDataKey:
        'CognitoIdentityServiceProvider.e33qfsolr15a3mvdb12******.*********.userData'
    },
    userConfirmed: false,
    userSub: '20716ced-34a0-45f0-aa9a-bff28b0b1f89'
}
```

### メールアドレスの検証は発生する？  
YES  
![](https://gyazo.com/7c4b83aa1bd7b4f3ad04e884a87d1134.png)

### SignUp直後のユーザーの状態は？
![](https://gyazo.com/e77173f13a75cd3620c41667f207d44d.png)


前提として、ユーザーのconfirmには2種類ある
？つまり即confirmすれば、メアド未認証でもユーザーはサインインできるわけね。
でもその場合、パスリセも聞かないしエイリアスとしてメールアドレスを持つこともできない。
CONFIRMを取り消すことはできない


メールアドレスをユーザー名として利用した場合には、どのようなサインアップになる？

Auth.signUp()は自己サインアップなの？管理者によるサインアップなの？


## Auth.confirmSignUp()
### Auth.confirmSignUp()の返り値は？  
`SUCCESS`のみ。


# ユーザープールへのソーシャル ID プロバイダーの追加
参考:  
https://docs.aws.amazon.com/ja_jp/cognito/latest/developerguide/cognito-user-pools-social-idp.html

facebook for developersからアプリIDを追加する。  
![](https://gyazo.com/ef1f4625a286d91e9b3e586db3830b46.png)

Cognito UserPoolにFacebookを設定し、かつアプリクライアントの設定として有効なIDプロバイダを設定する。  
![](https://gyazo.com/360bf513d0bfcb28fcc823fc0f1635e5.png)


# ユーザープールのソーシャルIdPのテスト
ログインURLにアクセスしてHosted UIを表示する。  
`https://******.auth.ap-northeast-1.amazoncognito.com/login?response_type=code&client_id=******&redirect_uri=https://example.com`  
![](https://gyazo.com/974e9bf84d9eaed31604f65fac2cfac3.png)

Facebookに遷移する。  
![](https://gyazo.com/45a543820c6adfcead038a56d001aa66.png)

SignIn後のクエリパラメータ  
https://example.com/?error_description=attributes+required%3A+%5Bemail%5D&error=invalid_request#_=_

## エラー
### FacebookアプリにCognitoUserPoolのドメインを登録してない場合    
![](https://gyazo.com/b2d389ed9b46a0c3a1c0197bc443dd0a.png)

### SignUp時の理由不明のエラー。  
リダイレクト先がhttps://example.comだからか？  
![](https://gyazo.com/3ab9f8bb055dac3de90288f0c036e2d4.png)
