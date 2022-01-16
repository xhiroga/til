import { Auth0Provider } from '@auth0/auth0-react';
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { getEnv } from './helpers';
import './index.css';
import reportWebVitals from './reportWebVitals';

ReactDOM.render(
  // @note -- どうもいきなりGoogle認証で入るとuserを取得しないような...設定が悪い？
  // @note -- ?code=****(&state=****) が返ってきた場合、勝手に吸収して /token に POSTする。
  <Auth0Provider
    domain={getEnv('REACT_APP_DOMAIN')}
    clientId={getEnv('REACT_APP_CLIENT_ID')}
    redirectUri={window.location.origin}
    audience={getEnv('REACT_APP_AUDIENCE')}
    scope='read:current_user update:current_user_metadata'
  >
    <React.StrictMode>
      <App />
    </React.StrictMode>
  </Auth0Provider>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
