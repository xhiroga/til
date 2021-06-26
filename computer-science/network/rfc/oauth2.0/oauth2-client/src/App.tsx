import * as querystring from 'querystring';
import React, { useCallback, useEffect, useState } from 'react';
import {
  BrowserRouter as Router,
  Route,
  Switch,
  useLocation,
} from 'react-router-dom';
import { Center } from './Center';
import { base64Encode, getEnv } from './helpers';

// @note host, origin, pathName などの用語の使い分けは https://developer.mozilla.org/ja/docs/Web/API/Location に準ずる。
const authorizationServerUrlOrigin = getEnv(
  'REACT_APP_AUTHORIZATION_SERVER_URL_ORIGIN'
);
const oauthClientId = getEnv('REACT_APP_OAUTH_CLIENT_ID');
const oauthClientPassword = getEnv('REACT_APP_OAUTH_CLIENT_PASSWORD');
// @note pathName は定義上 / を含む
const authorizationEndpointPathName = getEnv(
  'REACT_APP_AUTHORIZATION_ENDPOINT_PATH_NAME'
);
const tokenEndpointPathName = getEnv('REACT_APP_TOKEN_ENDPOINT_PATH_NAME');

const scope = getEnv('REACT_APP_ACCESS_TOKEN_SCOPE');

type AuthorizationResponseType = 'code' | 'token';
const authorizationResponseType: AuthorizationResponseType = 'code';

const redirectEndpointPath = 'callback';
const redirectEndpoint = `${window.location.protocol}//${window.location.host}/${redirectEndpointPath}`;

type AuthorizationRequest = {
  response_type: AuthorizationResponseType;
  client_id: string;
  redirect_uri?: string;
  scope?: string;
  state?: string;
};

type AuthorizationSuccessResponse = {
  code: string;
  state?: string;
};

type AuthorizationErrorResponse = {
  error: any;
  error_description?: string;
  error_uri?: string;
  state?: string;
};

type AuthorizationGrantType =
  | 'authorization_code'
  | 'implicit'
  | 'resource_owner_password_credential'
  | 'client_credential';
const authorizationGrantType: AuthorizationGrantType = 'authorization_code';

type AccessTokenRequest = {
  grant_type: AuthorizationGrantType;
  code: string;
  redirect_uri?: string; // 認可リクエスト時と同じ値でなければならない(codeの横取りを防ぐための仕様だと思われる)
  client_id?: string;
};

type AccessTokenSuccessResponse = {
  access_token: string;
  token_type?: string;
  expires_in: number;
  id_token?: string;
  refresh_token: string;
  tokenType: 'Bearer';
};

type AccessTokenErrorResponse = {
  error: any;
  error_description?: string;
  error_uri?: string;
};

const Home: React.FunctionComponent = () => {
  const authorizationRequest: AuthorizationRequest = {
    response_type: authorizationResponseType,
    client_id: oauthClientId,
    redirect_uri: redirectEndpoint,
  };
  var parameter = querystring.stringify(authorizationRequest);
  if (authorizationServerUrlOrigin.includes('cognito')) {
    parameter = parameter + `&scope=${scope}`; // + を URLエンコードしてはいけない
  }

  return (
    <Center>
      <a
        href={`${authorizationServerUrlOrigin}${authorizationEndpointPathName}?${parameter}`}
      >
        <button>Get OAuth Authorization Code</button>
      </a>
    </Center>
  );
};

const EndpointContent: React.FunctionComponent = () => {
  const { search } = useLocation();
  const [authorizationCode, setAuthorizationCode] = useState<string | null>(
    null
  );
  const [tokenEndpointResponse, setTokenEndpointResponse] =
    useState<AccessTokenSuccessResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const queries = querystring.parse(search.replace(`?`, ``));
    if (queries.error !== undefined) {
      setErrorMessage(JSON.stringify(queries as AuthorizationErrorResponse));
    }
    setAuthorizationCode((queries as AuthorizationSuccessResponse).code);
    setErrorMessage(null);

    return () => {
      setAuthorizationCode(null);
      setErrorMessage(null);
    };
  }, [search, setAuthorizationCode, setErrorMessage]);

  const onClick = useCallback(() => {
    if (authorizationCode === null) {
      return;
    }
    const accessTokenRequest: AccessTokenRequest = {
      grant_type: authorizationGrantType,
      code: authorizationCode,
      redirect_uri: redirectEndpoint,
      client_id: oauthClientId,
    };
    const params = querystring.stringify(accessTokenRequest);
    const headers = {
      'Content-Type': `application/x-www-form-urlencoded`,
      Authorization: `Basic ${base64Encode(
        `${oauthClientId}:${oauthClientPassword}`
      )}`,
    };
    void fetch(
      // https://docs.aws.amazon.com/ja_jp/cognito/latest/developerguide/token-endpoint.html
      `${authorizationServerUrlOrigin}${tokenEndpointPathName}?${params}`,
      {
        method: `POST`,
        headers: headers,
      }
    )
      .then((response) => response.json())
      .then((response) => {
        if (response.error !== undefined) {
          setErrorMessage(JSON.stringify(response as AccessTokenErrorResponse));
        } else {
          setTokenEndpointResponse(response as AccessTokenSuccessResponse);
          setErrorMessage(null);
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
          {errorMessage && <p style={{ color: `#FF0000` }}>{errorMessage}</p>}
        </div>
      )}
      <div>
        <a href={`/`}>
          <button>ホームに戻る</button>
        </a>
      </div>
    </Center>
  );
};

const App: React.FunctionComponent = () => {
  return (
    <Router>
      <Switch>
        <Route path={`/${redirectEndpointPath}`}>
          <EndpointContent />
        </Route>
        <Route path={`/`}>
          <Home />
        </Route>
      </Switch>
    </Router>
  );
};

export default App;
