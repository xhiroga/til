import * as querystring from 'querystring';
import React, { useCallback, useEffect, useState } from 'react';
import {
  BrowserRouter as Router,
  Route,
  Switch,
  useLocation,
} from 'react-router-dom';
import './App.css';
import Center from './Center';

const cognitoUserPoolOAuthHost = process.env.REACT_APP_COGNITO_USER_POOL_OAUTH_HOST;
const cognitoUserPoolClientId = process.env.REACT_APP_COGNITO_USER_POOL_CLIENT_ID;
const cognitoUserPoolClientSecret = process.env.REACT_APP_COGNITO_USER_POOL_CLIENT_SECRET;
const callbackPath = `callback`;
const redirectUri = `${window.location.protocol}//${window.location.host}/${callbackPath}`;
type errors =
  | `invalid_client` // アプリクライアントが存在しない、IDが誤っている、BasicじゃなくてBearerを使っていた、など
  | `invalid_grant`; // Authorizationトークンが失効している（Refreshトークンの失効でもおそらく）

const base64Encode = (value: string) => Buffer.from(value).toString('base64');

function Start() {
  return (
    <Center>
      <a
        href={`${cognitoUserPoolOAuthHost}/login?response_type=code&client_id=${cognitoUserPoolClientId}&scope=aws.cognito.signin.user.admin+openid+profile&redirect_uri=${redirectUri}`}
      >
        <button>Get OAuth Authorization Code</button>
      </a>
    </Center>
  );
}

type TokenEndpointResponse = {
  accessToken: string;
  expiresIn: number;
  idToken: string;
  refreshToken: string;
  tokenType: 'Bearer';
};

function Callback() {
  const { search } = useLocation();
  const [authorizationCode, setAuthorizationCode] = useState<string | null>(
    null
  );
  const [tokenEndpointResponse, setTokenEndpointResponse] =
    useState<TokenEndpointResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  useEffect(() => {
    const queries = querystring.parse(search.replace(`?`, ``));
    if (queries.code !== undefined) {
      setAuthorizationCode(queries.code as string);
    }
    return () => {
      setAuthorizationCode(null);
    };
  }, [search, setAuthorizationCode]);

  const onClick = useCallback(() => {
    if (authorizationCode === null) {
      return;
    }
    void fetch(
      // https://docs.aws.amazon.com/ja_jp/cognito/latest/developerguide/token-endpoint.html
      `${cognitoUserPoolOAuthHost}/oauth2/token?grant_type=authorization_code&client_id=${cognitoUserPoolClientId}&code=${authorizationCode}&redirect_uri=${redirectUri}`,
      {
        method: `POST`,
        headers: {
          'Content-Type': `application/x-www-form-urlencoded`,
          Authorization: `Basic ${base64Encode(
            `${cognitoUserPoolClientId}:${cognitoUserPoolClientSecret}`
          )}`,
        },
      }
    )
      .then((response) => response.json())
      .then((response) => {
        if (response.error) {
          setErrorMessage(JSON.stringify(response.error as errors));
        } else {
          setTokenEndpointResponse({
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            idToken: response.id_token,
            expiresIn: response.expires_in,
            tokenType: response.token_type,
          });
        }
      });
  }, [authorizationCode, setTokenEndpointResponse, setErrorMessage]);

  return (
    <Center>
      {authorizationCode === null ? (
        <p>No authorization code was found.</p>
      ) : (
        <div>
          {tokenEndpointResponse ? (
            <>
              <p>Tokens are...</p>
              <p
                style={{
                  maxHeight: '600px',
                  maxWidth: '800px',
                  wordWrap: 'break-word',
                  overflow: 'scroll',
                }}
              >
                {JSON.stringify(tokenEndpointResponse, null, 4)}
              </p>
            </>
          ) : (
            <>
              <p>{`Authorization code ${authorizationCode} was found.\nNext, get access token.`}</p>
              <button onClick={onClick}>Get access token</button>
            </>
          )}
          {errorMessage && (
            <p style={{ color: `#FF0000` }}>{`error: ${errorMessage}`}</p>
          )}
        </div>
      )}
      <div>
        <a href={`/`}>
          <button>初期画面に戻る</button>
        </a>
      </div>
    </Center>
  );
}

function App() {
  return (
    <Router>
      <Switch>
        <Route path={`/${callbackPath}`}>
          <Callback />
        </Route>
        <Route path='/'>
          <Start />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
